from __future__ import annotations

from dataclasses import asdict
import logging
import math
from pathlib import Path
import re
from statistics import median
from typing import Iterable, Optional

from ..ann import HnswIndex
from ..compression import CodexExecCompressionBackend
from ..config import EngineTuning, MemoryConfig
from ..maintenance import AnnMaintenanceCoordinator
from ..embeddings import (
    cosine_similarity,
    pack_embedding,
    try_create_default_embedding_backend,
    unpack_embedding,
)
from ..models import Anchor, Fragment, FragmentFeedback, FragmentReference
from ..store import SQLiteStore
from .ports import AnnIndex, CompressionBackend, EmbeddingBackend, Store
from .types import (
    FeedbackRequest,
    FeedbackResult,
    FetchRequest,
    FragmentView,
    ReadActiveCollaborationPolicyRequest,
    ReadActiveCollaborationPolicyResponse,
    RecentRequest,
    RememberRequest,
    SearchRequest,
    SearchItemView,
    SearchResponse,
    StatsResult,
)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


LEXICAL_SEPARATOR_PATTERN = re.compile(r"[`'\"/\\._\-,:;!?()[\]{}<>|*+=~@#$%^&]+|[、。・「」『』【】（）！？：；　]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
SEARCH_TERM_PATTERN = re.compile(r"[0-9a-z]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+")
COLLABORATION_POLICY_KIND = "collaboration_policy"
LOGGER = logging.getLogger(__name__)


class SearchStatusError(RuntimeError):
    def __init__(self, status: dict):
        super().__init__(status.get("code", "search_status"))
        self.status = status


class BreathingMemoryEngine:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        tuning: Optional[EngineTuning] = None,
        store: Optional[Store] = None,
        compression_backend: Optional[CompressionBackend] = None,
        embedding_backend: Optional[EmbeddingBackend] = None,
        ann_index: Optional[AnnIndex] = None,
    ):
        self.config = config or MemoryConfig()
        self.tuning = tuning or EngineTuning()
        self._validate_retrieval_mode(self.config.retrieval_mode)
        self.store = store or SQLiteStore(Path(self.config.db_path))
        self.compression_backend = compression_backend or CodexExecCompressionBackend()
        self.embedding_backend = embedding_backend or try_create_default_embedding_backend()
        self.ann_index = ann_index or HnswIndex(Path(self.config.db_path))
        self._ann_maintenance = AnnMaintenanceCoordinator(Path(self.config.db_path))

    def close(self) -> None:
        self.store.close()

    def remember(self, request: RememberRequest) -> FragmentView:
        return self._build_fragment_view(
            self._remember_fragment(
                content=request.content,
                actor=request.actor,
                reply_to=request.reply_to,
                source_fragment_ids=request.source_fragment_ids,
                kind=request.kind,
            )
        )

    def _remember_fragment(
        self,
        *,
        content: str,
        actor: str,
        reply_to: Optional[int],
        source_fragment_ids: Optional[Iterable[int]],
        kind: Optional[str],
    ) -> Fragment:
        if actor not in {"user", "agent"}:
            raise ValueError("actor must be 'user' or 'agent'")
        if not content or not content.strip():
            raise ValueError("content must not be empty")
        if reply_to is not None and not self.store.anchor_exists(reply_to):
            raise ValueError("reply_to anchor not found")

        normalized_kind = self._normalize_kind(kind)
        normalized_sources = self._normalize_source_fragment_ids(source_fragment_ids)
        self._validate_source_fragment_ids(normalized_sources)

        duplicate_fragment = self._find_duplicate_agent_fragment(
            actor=actor,
            content=content,
            reply_to=reply_to,
            kind=normalized_kind,
        )
        if duplicate_fragment is not None:
            self._merge_material_references(
                from_anchor_id=duplicate_fragment.anchor_id,
                source_fragment_ids=normalized_sources,
            )
            self._apply_observation_driven_promotion(normalized_sources)
            return duplicate_fragment

        anchor_id = self.store.create_anchor(
            replies_to_anchor_id=reply_to,
            is_root=reply_to is None,
        )
        fragment_id = self.store.create_fragment(
            anchor_id=anchor_id,
            parent_id=None,
            actor=actor,
            kind=normalized_kind,
            content=content,
            embedding_vector=self._embed_content(content),
            layer="working",
        )
        self.store.create_reference(from_anchor_id=anchor_id, fragment_id=fragment_id)
        self.store.create_feedback(from_anchor_id=anchor_id, fragment_id=fragment_id, verdict="positive")

        for source_fragment_id in normalized_sources:
            self.store.create_reference(from_anchor_id=anchor_id, fragment_id=source_fragment_id)

        self._apply_observation_driven_promotion(normalized_sources)
        counters = {"compress": 0, "delete": 0}
        self._stabilize_after_insert(
            current_anchor_id=anchor_id,
            counters=counters,
            protected_fragment_ids={fragment_id},
        )
        self._record_sequence_metrics(anchor_id=anchor_id, counters=counters)
        self._sync_ann_index_for_new_fragment(fragment_id)
        fragment = self.store.get_fragment(fragment_id)
        if fragment is None:
            raise RuntimeError("remembered fragment was not preserved")
        return fragment

    def recent(self, request: RecentRequest) -> SearchResponse:
        return self._recent_response(
            limit=request.limit,
            actor=request.actor,
            reply_to=request.reply_to,
        )

    def _recent_response(
        self,
        *,
        limit: int = 4,
        actor: Optional[str] = None,
        reply_to: Optional[int] = None,
    ) -> SearchResponse:
        normalized_limit = int(limit)
        if normalized_limit <= 0:
            raise ValueError("limit must be a positive integer")
        if actor is not None and actor not in {"user", "agent"}:
            raise ValueError("actor must be 'user' or 'agent'")
        if reply_to is not None and not self.store.anchor_exists(reply_to):
            raise ValueError("reply_to anchor not found")

        fragments = self.store.list_recent_root_fragments(
            limit=normalized_limit,
            actor=actor,
            reply_to_anchor_id=reply_to,
        )
        return SearchResponse(
            items=[self._build_search_item_view(fragment) for fragment in fragments],
            count=len(fragments),
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        return self._search_response(
            query=request.query,
            result_count=request.result_count,
            search_effort=request.search_effort,
            actor=request.actor,
            kind=request.kind,
            include_diagnostics=request.include_diagnostics,
        )

    def _search_response(
        self,
        query: str,
        result_count: Optional[int] = None,
        search_effort: Optional[int] = None,
        actor: Optional[str] = None,
        kind: Optional[str] = None,
        include_diagnostics: bool = False,
    ) -> SearchResponse:
        normalized_result_count = self._normalize_result_count(result_count)
        normalized_search_effort = self._normalize_search_effort(search_effort)
        normalized_actor = self._normalize_actor_filter(actor)
        normalized_kind = self._normalize_kind(kind)
        retrieval_mode = self._resolve_retrieval_mode()

        normalized_query = self._normalize_lexical_text(query)
        query_terms = self._search_terms(normalized_query)
        if not normalized_query:
            return self._search_by_search_priority(
                normalized_result_count,
                actor=normalized_actor,
                kind=normalized_kind,
                retrieval_mode=retrieval_mode,
                include_diagnostics=include_diagnostics,
            )

        if retrieval_mode == "default":
            return self._semantic_search(
                query,
                normalized_result_count,
                actor=normalized_actor,
                kind=normalized_kind,
                retrieval_mode="default",
                search_effort=normalized_search_effort,
                include_diagnostics=include_diagnostics,
            )
        if retrieval_mode == "lite":
            return self._semantic_search(
                query,
                normalized_result_count,
                actor=normalized_actor,
                kind=normalized_kind,
                retrieval_mode="lite",
                search_effort=normalized_search_effort,
                include_diagnostics=include_diagnostics,
            )

        fragments = self.store.list_fragments(actor=normalized_actor, kind=normalized_kind)
        lexical_matches: list[tuple[Fragment, tuple[int, int, int, float]]] = []
        if normalized_query:
            for fragment in fragments:
                lexical_rank = self._lexical_rank(fragment.content, normalized_query, query_terms)
                if lexical_rank is None:
                    continue
                lexical_matches.append((fragment, lexical_rank))
        else:
            lexical_matches = [(fragment, (0, 0, 0, 0.0)) for fragment in fragments]
        lexical_matches.sort(
            key=lambda item: (
                -item[1][0],
                -item[1][1],
                -item[1][2],
                -item[1][3],
                -self._search_priority(item[0].id),
                item[0].id,
            )
        )
        items = [
            self._build_search_item_view(
                fragment,
                diagnostics=self._lexical_diagnostics(lexical_rank, retrieval_mode)
                if include_diagnostics
                else None,
            )
            for fragment, lexical_rank in lexical_matches[:normalized_result_count]
        ]
        return SearchResponse(items=items, count=len(items))

    def read_active_collaboration_policy(
        self,
        request: ReadActiveCollaborationPolicyRequest,
    ) -> ReadActiveCollaborationPolicyResponse:
        return self._read_active_collaboration_policy_response(token_budget=request.token_budget)

    def _read_active_collaboration_policy_response(
        self,
        *,
        token_budget: Optional[int] = None,
    ) -> ReadActiveCollaborationPolicyResponse:
        normalized_token_budget = self._normalize_token_budget(token_budget)
        fragments = self.store.list_fragments(kind=COLLABORATION_POLICY_KIND)
        ordered_fragments = self._sort_fragments_by_search_priority(fragments)
        if not ordered_fragments:
            return ReadActiveCollaborationPolicyResponse(
                items=[],
                count=0,
                token_budget=normalized_token_budget,
                used_token_budget=0,
                truncated=False,
            )

        items: list[SearchItemView] = []
        used_token_budget = 0
        truncated = False
        for fragment in ordered_fragments:
            estimated_tokens = self._estimate_token_count(fragment.content)
            if items and used_token_budget + estimated_tokens > normalized_token_budget:
                truncated = True
                break
            if not items and estimated_tokens > normalized_token_budget:
                items.append(self._build_search_item_view(fragment))
                used_token_budget += estimated_tokens
                truncated = len(ordered_fragments) > 1
                break
            items.append(self._build_search_item_view(fragment))
            used_token_budget += estimated_tokens

        return ReadActiveCollaborationPolicyResponse(
            items=items,
            count=len(items),
            token_budget=normalized_token_budget,
            used_token_budget=used_token_budget,
            truncated=truncated,
        )

    def fetch(self, request: FetchRequest) -> SearchResponse:
        return self._fetch_response(
            fragment_id=request.fragment_id,
            anchor_id=request.anchor_id,
        )

    def _fetch_response(
        self,
        fragment_id: Optional[int] = None,
        anchor_id: Optional[int] = None,
    ) -> SearchResponse:
        if (fragment_id is None) == (anchor_id is None):
            raise ValueError("exactly one of fragment_id or anchor_id is required")
        if fragment_id is not None:
            fragment = self.store.get_fragment(fragment_id)
            if fragment is None:
                return SearchResponse(items=[], count=0)
            return SearchResponse(items=[self._build_search_item_view(fragment)], count=1)

        assert anchor_id is not None
        fragments = self.store.list_fragments_by_anchor(anchor_id)
        fragments.sort(key=lambda fragment: (-self._search_priority(fragment.id), fragment.id))
        return SearchResponse(
            items=[self._build_search_item_view(fragment) for fragment in fragments],
            count=len(fragments),
        )

    def feedback(self, request: FeedbackRequest) -> FeedbackResult:
        return self._feedback_result(
            from_anchor_id=request.from_anchor_id,
            fragment_id=request.fragment_id,
            verdict=request.verdict,
        )

    def _feedback_result(
        self,
        *,
        from_anchor_id: int,
        fragment_id: int,
        verdict: str,
    ) -> FeedbackResult:
        if verdict not in {"positive", "neutral", "negative"}:
            raise ValueError("verdict must be 'positive', 'neutral', or 'negative'")
        if not self.store.anchor_exists(from_anchor_id):
            raise ValueError("from_anchor_id not found")
        if not self.store.fragment_exists(fragment_id):
            raise ValueError("fragment not found")
        self.store.create_feedback(
            from_anchor_id=from_anchor_id,
            fragment_id=fragment_id,
            verdict=verdict,
        )
        return FeedbackResult(
            fragment_id=fragment_id,
            verdict=verdict,
            confidence_score=self._confidence_score(fragment_id),
            search_priority=self._search_priority(fragment_id),
        )

    def stats(self) -> StatsResult:
        return self._stats_result()

    def _stats_result(self) -> StatsResult:
        fragments = self.store.list_fragments()
        budgets = self._budget_snapshot()
        usage = self._layer_usage_snapshot()
        recent_metrics = self._recent_sequence_metrics()
        parameters = asdict(self.tuning)
        parameters["retrieval_mode"] = self.config.retrieval_mode
        parameters["total_capacity_mb"] = self.config.total_capacity_mb
        parameters["total_capacity"] = self._total_capacity_bytes()
        return StatsResult(
            fragment_count=len(fragments),
            working_count=len([fragment for fragment in fragments if fragment.layer == "working"]),
            holding_count=len([fragment for fragment in fragments if fragment.layer == "holding"]),
            working_usage=usage["working"],
            holding_usage=usage["holding"],
            working_budget=budgets["working_budget"],
            holding_budget=budgets["holding_budget"],
            working_ratio=budgets["working_ratio"],
            recent_compress_count=sum(metric.compress_count for metric in recent_metrics),
            recent_delete_count=sum(metric.delete_count for metric in recent_metrics),
            parameters=parameters,
        )

    def export_fragment_graph(self) -> list[SearchItemView]:
        return [
            self._build_search_item_view(fragment)
            for fragment in self.store.list_fragments()
        ]

    def warmup_embeddings(self) -> bool:
        if self.embedding_backend is None:
            return False
        self.embedding_backend.warmup()
        return True

    def start_background_embedding_warmup(self) -> bool:
        if self.embedding_backend is None:
            return False
        return self.embedding_backend.start_background_warmup(on_error=self._handle_background_warmup_error)

    def _validate_retrieval_mode(self, retrieval_mode: str) -> None:
        if retrieval_mode not in {"auto", "super_lite", "lite", "default"}:
            raise ValueError("retrieval_mode must be 'auto', 'super_lite', 'lite', or 'default'")

    def _resolve_retrieval_mode(self) -> str:
        if self.config.retrieval_mode == "auto":
            if self.embedding_backend is None:
                return "super_lite"
            if self.ann_index.support_available():
                return "default"
            return "lite"
        return self.config.retrieval_mode

    def _semantic_search(
        self,
        query: str,
        result_count: int,
        *,
        actor: Optional[str],
        kind: Optional[str],
        retrieval_mode: str,
        search_effort: int,
        include_diagnostics: bool = False,
    ) -> SearchResponse:
        if self.embedding_backend is None:
            raise RuntimeError(f"retrieval mode '{retrieval_mode}' requires an embedding backend")
        fragments = self.store.list_fragments(actor=actor, kind=kind)
        if not fragments:
            return SearchResponse(items=[], count=0)

        try:
            self._ensure_fragment_embeddings(
                fragments,
                timeout_ms=self.tuning.ann_wait_timeout_ms,
                current_mode="lite" if retrieval_mode == "default" else retrieval_mode,
            )
        except SearchStatusError as exc:
            return self._status_response(exc.status)
        embedded_fragments = [
            fragment
            for fragment in self.store.list_fragments(actor=actor, kind=kind)
            if fragment.embedding_vector is not None
        ]
        if not embedded_fragments:
            return SearchResponse(items=[], count=0)

        query_vector = self.embedding_backend.embed_texts([query])[0]
        try:
            if retrieval_mode == "default":
                semantic_candidates = self._default_semantic_candidates(
                    embedded_fragments=embedded_fragments,
                    query_vector=query_vector,
                    result_count=result_count,
                    search_effort=search_effort,
                )
            else:
                semantic_candidates = self._lite_semantic_candidates(
                    embedded_fragments=embedded_fragments,
                    query_vector=query_vector,
                    result_count=result_count,
                )
        except SearchStatusError as exc:
            return self._status_response(exc.status)
        return self._rerank_semantic_candidates(
            semantic_candidates=semantic_candidates,
            retrieval_mode=retrieval_mode,
            include_diagnostics=include_diagnostics,
        )

    def _lite_semantic_candidates(
        self,
        *,
        embedded_fragments: list[Fragment],
        query_vector: list[float],
        result_count: int,
    ) -> list[tuple[Fragment, float]]:
        semantic_candidates: list[tuple[Fragment, float]] = []
        for fragment in embedded_fragments:
            assert fragment.embedding_vector is not None
            similarity = cosine_similarity(query_vector, unpack_embedding(fragment.embedding_vector))
            semantic_similarity = (similarity + 1.0) / 2.0
            semantic_candidates.append((fragment, semantic_similarity))
        semantic_candidates.sort(
            key=lambda item: (
                -item[1],
                -self._search_priority(item[0].id),
                item[0].id,
            )
        )
        return semantic_candidates[:result_count]

    def _default_semantic_candidates(
        self,
        *,
        embedded_fragments: list[Fragment],
        query_vector: list[float],
        result_count: int,
        search_effort: int,
    ) -> list[tuple[Fragment, float]]:
        if not self.ann_index.support_available():
            raise RuntimeError("retrieval mode 'default' requires HNSW index support")
        vectors_by_fragment_id = {
            fragment.id: unpack_embedding(fragment.embedding_vector)
            for fragment in embedded_fragments
            if fragment.embedding_vector is not None
        }
        self._ensure_ann_ready_for_search(vectors_by_fragment_id)
        fragments_by_id = {fragment.id: fragment for fragment in embedded_fragments}
        semantic_candidates: list[tuple[Fragment, float]] = []
        with self._ann_maintenance.acquire_shared(timeout_ms=self.tuning.ann_wait_timeout_ms) as acquired:
            if not acquired:
                raise SearchStatusError(self._ann_maintenance.read_status(current_mode="lite"))
            for fragment_id in self.ann_index.query(
                vector=query_vector,
                limit=result_count,
                search_effort=search_effort,
            ):
                fragment = fragments_by_id.get(fragment_id)
                if fragment is None or fragment.embedding_vector is None:
                    continue
                similarity = cosine_similarity(query_vector, unpack_embedding(fragment.embedding_vector))
                semantic_similarity = (similarity + 1.0) / 2.0
                semantic_candidates.append((fragment, semantic_similarity))
        return semantic_candidates

    def _rerank_semantic_candidates(
        self,
        *,
        semantic_candidates: list[tuple[Fragment, float]],
        retrieval_mode: str,
        include_diagnostics: bool,
    ) -> SearchResponse:
        normalized_priorities = self._normalize_search_priorities(
            [self._search_priority(fragment.id) for fragment, _ in semantic_candidates]
        )
        reranked: list[tuple[Fragment, float, float, float]] = []
        for (fragment, similarity), normalized_priority in zip(semantic_candidates, normalized_priorities):
            final_score = similarity * normalized_priority
            reranked.append((fragment, final_score, similarity, normalized_priority))
        reranked.sort(
            key=lambda item: (
                -item[1],
                -item[2],
                -self._search_priority(item[0].id),
                item[0].id,
            )
        )
        items = [
            self._build_search_item_view(
                fragment,
                diagnostics={
                    "retrieval_mode": retrieval_mode,
                    "semantic_similarity": similarity,
                    "normalized_priority": normalized_priority,
                    "ranking_score": final_score,
                }
                if include_diagnostics
                else None,
            )
            for fragment, final_score, similarity, normalized_priority in reranked
        ]
        return SearchResponse(items=items, count=len(items))

    def _search_by_search_priority(
        self,
        result_count: int,
        *,
        actor: Optional[str],
        kind: Optional[str],
        retrieval_mode: str,
        include_diagnostics: bool = False,
    ) -> SearchResponse:
        fragments = self._sort_fragments_by_search_priority(
            self.store.list_fragments(actor=actor, kind=kind)
        )
        items = [
            self._build_search_item_view(
                fragment,
                diagnostics={
                    "retrieval_mode": retrieval_mode,
                    "ranking_score": self._search_priority(fragment.id),
                }
                if include_diagnostics
                else None,
            )
            for fragment in fragments[:result_count]
        ]
        return SearchResponse(items=items, count=len(items))

    def _normalize_result_count(self, result_count: Optional[int]) -> int:
        value = self.tuning.default_result_count if result_count is None else int(result_count)
        if value < self.tuning.default_result_count or value % self.tuning.default_result_count != 0:
            raise ValueError("result_count must be 4 * 2^n")
        quotient = value // self.tuning.default_result_count
        if quotient & (quotient - 1):
            raise ValueError("result_count must be 4 * 2^n")
        return value

    def _normalize_search_effort(self, search_effort: Optional[int]) -> int:
        value = self.tuning.default_search_effort if search_effort is None else int(search_effort)
        if value < self.tuning.default_search_effort or value % self.tuning.default_search_effort != 0:
            raise ValueError("search_effort must be 32 * 2^n")
        quotient = value // self.tuning.default_search_effort
        if quotient & (quotient - 1):
            raise ValueError("search_effort must be 32 * 2^n")
        return value

    def _normalize_token_budget(self, token_budget: Optional[int]) -> int:
        value = 512 if token_budget is None else int(token_budget)
        if value <= 0:
            raise ValueError("token_budget must be a positive integer")
        return value

    def _normalize_actor_filter(self, actor: Optional[str]) -> Optional[str]:
        if actor is None:
            return None
        if actor not in {"user", "agent"}:
            raise ValueError("actor must be 'user' or 'agent'")
        return actor

    def _normalize_source_fragment_ids(self, source_fragment_ids: Optional[Iterable[int]]) -> list[int]:
        if source_fragment_ids is None:
            return []
        normalized: list[int] = []
        seen: set[int] = set()
        for fragment_id in source_fragment_ids:
            value = int(fragment_id)
            if value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _normalize_dedup_content(self, content: str) -> str:
        return content.replace("\r\n", "\n").replace("\r", "\n").strip()

    def _normalize_kind(self, kind: Optional[str]) -> Optional[str]:
        if kind is None:
            return None
        normalized = str(kind).strip()
        if not normalized:
            raise ValueError("kind must not be blank")
        return normalized

    def _embed_content(self, content: str) -> Optional[bytes]:
        if self.embedding_backend is None:
            return None
        vector = self.embedding_backend.embed_texts([content])[0]
        return pack_embedding(vector)

    def _handle_background_warmup_error(self, exc: Exception) -> None:
        LOGGER.warning(
            "Background embedding warmup failed; semantic calls will retry on demand.",
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    def _ensure_fragment_embeddings(
        self,
        fragments: Iterable[Fragment],
        *,
        timeout_ms: int | None = None,
        current_mode: str = "lite",
    ) -> None:
        if self.embedding_backend is None:
            return
        missing = [fragment for fragment in fragments if fragment.embedding_vector is None]
        if not missing:
            return
        with self._ann_maintenance.acquire_exclusive(timeout_ms=timeout_ms) as acquired:
            if not acquired:
                raise SearchStatusError(self._ann_maintenance.read_status(current_mode=current_mode))
            current_fragments = {fragment.id: fragment for fragment in self.store.list_fragments()}
            current_missing = [
                current_fragments[fragment.id]
                for fragment in missing
                if fragment.id in current_fragments and current_fragments[fragment.id].embedding_vector is None
            ]
            if not current_missing:
                return
            with self._ann_maintenance.active_status(
                code="rebuild_in_progress",
                phase="embedding_backfill",
                fragment_count=len(current_missing),
                current_mode=current_mode,
                suggested_action="retry",
            ):
                vectors = self.embedding_backend.embed_texts([fragment.content for fragment in current_missing])
                for fragment, vector in zip(current_missing, vectors):
                    self.store.update_fragment_embedding(fragment.id, pack_embedding(vector))
                self._rebuild_ann_index_locked(self.store.list_fragments())

    def _normalize_search_priorities(self, priorities: list[float]) -> list[float]:
        if not priorities:
            return []
        lower = min(priorities)
        upper = max(priorities)
        if math.isclose(lower, upper):
            return [1.0 for _ in priorities]
        span = upper - lower
        return [0.1 + 0.9 * ((priority - lower) / span) for priority in priorities]

    def _find_duplicate_agent_fragment(
        self,
        actor: str,
        content: str,
        reply_to: Optional[int],
        kind: Optional[str],
    ) -> Optional[Fragment]:
        if actor != "agent" or reply_to is None:
            return None

        normalized_content = self._normalize_dedup_content(content)
        for candidate in self.store.list_root_fragments_replying_to_anchor(
            reply_to,
            actor=actor,
            kind=kind,
        ):
            if self._normalize_dedup_content(candidate.content) == normalized_content:
                return candidate
        return None

    def _merge_material_references(
        self,
        from_anchor_id: int,
        source_fragment_ids: Iterable[int],
    ) -> None:
        existing_fragment_ids = {
            reference.fragment_id
            for reference in self.store.list_references_from_anchor(from_anchor_id)
        }
        for fragment_id in source_fragment_ids:
            if fragment_id in existing_fragment_ids:
                continue
            self.store.create_reference(from_anchor_id=from_anchor_id, fragment_id=fragment_id)
            existing_fragment_ids.add(fragment_id)

    def _normalize_lexical_text(self, text: str) -> str:
        normalized = LEXICAL_SEPARATOR_PATTERN.sub(" ", text.casefold())
        return WHITESPACE_PATTERN.sub(" ", normalized).strip()

    def _search_terms(self, normalized_text: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for term in SEARCH_TERM_PATTERN.findall(normalized_text):
            if term in seen:
                continue
            seen.add(term)
            terms.append(term)
        return terms

    def _lexical_rank(
        self,
        content: str,
        normalized_query: str,
        query_terms: list[str],
    ) -> tuple[int, int, int, float] | None:
        normalized_content = self._normalize_lexical_text(content)
        exact_match = int(bool(normalized_query) and normalized_query in normalized_content)
        if not query_terms:
            return (exact_match, exact_match, 0, 0.0) if exact_match or not normalized_query else None

        matched_terms = sum(1 for term in query_terms if term in normalized_content)
        if matched_terms == 0 and not exact_match:
            return None
        all_terms_present = int(matched_terms == len(query_terms))
        coverage = matched_terms / len(query_terms)
        return (all_terms_present, exact_match, matched_terms, coverage)

    def _validate_source_fragment_ids(self, source_fragment_ids: list[int]) -> None:
        for fragment_id in source_fragment_ids:
            if not self.store.fragment_exists(fragment_id):
                raise ValueError(f"source fragment not found: {fragment_id}")

    def _apply_observation_driven_promotion(self, source_fragment_ids: Iterable[int]) -> None:
        for fragment_id in source_fragment_ids:
            fragment = self.store.get_fragment(fragment_id)
            if fragment is None or fragment.layer != "holding":
                continue
            self.store.update_fragment_layer(fragment_id, "working")

    def _stabilize_after_insert(
        self,
        current_anchor_id: int,
        counters: dict[str, int],
        protected_fragment_ids: set[int],
    ) -> None:
        del current_anchor_id
        while True:
            if self._layer_usage_exceeds_budget("working"):
                if not self._compress_once(counters, protected_fragment_ids):
                    raise RuntimeError("unable to free working capacity")
                continue
            if self._layer_usage_exceeds_budget("holding"):
                if not self._delete_once(counters, protected_fragment_ids):
                    raise RuntimeError("unable to free holding capacity")
                continue
            return

    def _compress_once(self, counters: dict[str, int], protected_fragment_ids: set[int]) -> bool:
        attempted_fragment_ids: set[int] = set()
        while True:
            candidate = self._select_compression_candidate(protected_fragment_ids | attempted_fragment_ids)
            if candidate is None:
                return False

            result = self.compression_backend.compress(
                candidate.content,
                self.tuning.compression_ratio,
            )
            if not result.content.strip() or self._content_size(result.content) >= candidate.content_length:
                self.store.increment_compression_fail_count(candidate.id)
                attempted_fragment_ids.add(candidate.id)
                continue

            while self._layer_usage_exceeds_budget("holding", extra_bytes=candidate.content_length):
                if not self._delete_once(counters, protected_fragment_ids=set()):
                    self.store.increment_compression_fail_count(candidate.id)
                    attempted_fragment_ids.add(candidate.id)
                    break
            else:
                self._materialize_compressed_child(candidate=candidate, content=result.content)
                counters["compress"] += 1
                return True

    def _materialize_compressed_child(self, *, candidate: Fragment, content: str) -> int:
        child_id = self.store.create_fragment(
            anchor_id=candidate.anchor_id,
            parent_id=candidate.id,
            actor=candidate.actor,
            kind=candidate.kind,
            content=content,
            embedding_vector=self._embed_content(content),
            layer="working",
        )
        self.store.copy_references(candidate.id, child_id)
        self.store.copy_feedback(candidate.id, child_id)
        self.store.create_reference(from_anchor_id=candidate.anchor_id, fragment_id=child_id)
        self.store.update_fragment_layer(candidate.id, "holding")
        self._sync_ann_index_for_new_fragment(child_id)
        return child_id

    def _delete_once(self, counters: dict[str, int], protected_fragment_ids: set[int]) -> bool:
        candidate = self._select_delete_candidate(protected_fragment_ids)
        if candidate is None:
            return False
        self.store.delete_fragment(candidate.id)
        self._remove_from_ann_index(candidate.id)
        counters["delete"] += 1
        return True

    def _sync_ann_index_for_new_fragment(self, fragment_id: int) -> None:
        if self.embedding_backend is None or not self.ann_index.support_available():
            return
        with self._ann_maintenance.acquire_exclusive() as acquired:
            if not acquired:  # pragma: no cover
                return
            fragment = self.store.get_fragment(fragment_id)
            if fragment is None or fragment.embedding_vector is None:
                return
            fragments = self.store.list_fragments()
            status = self.ann_index.inspect(
                fragment_ids=[item.id for item in fragments if item.id != fragment_id],
                embedding_model=self.config.embedding_model,
            )
            if not status.get("ready"):
                self._rebuild_ann_index_locked(fragments)
                return
            try:
                self.ann_index.append(
                    fragment_id=fragment.id,
                    vector=unpack_embedding(fragment.embedding_vector),
                    embedding_model=self.config.embedding_model,
                )
            except Exception:
                self._rebuild_ann_index_locked(fragments)

    def _remove_from_ann_index(self, fragment_id: int) -> None:
        if self.embedding_backend is None or not self.ann_index.support_available():
            return
        with self._ann_maintenance.acquire_exclusive() as acquired:
            if not acquired:  # pragma: no cover
                return
            try:
                self.ann_index.remove(fragment_id)
            except Exception:
                self._rebuild_ann_index_locked()

    def _rebuild_ann_index(self, fragments: Optional[list[Fragment]] = None) -> None:
        if self.embedding_backend is None or not self.ann_index.support_available():
            return
        with self._ann_maintenance.acquire_exclusive() as acquired:
            if not acquired:  # pragma: no cover
                return
            self._rebuild_ann_index_locked(fragments)

    def _rebuild_ann_index_locked(self, fragments: Optional[list[Fragment]] = None) -> None:
        if self.embedding_backend is None or not self.ann_index.support_available():
            return
        current_fragments = self.store.list_fragments() if fragments is None else fragments
        vectors_by_fragment_id = {
            fragment.id: unpack_embedding(fragment.embedding_vector)
            for fragment in current_fragments
            if fragment.embedding_vector is not None
        }
        self.ann_index.rebuild(
            vectors_by_fragment_id=vectors_by_fragment_id,
            embedding_model=self.config.embedding_model,
        )

    def _ensure_ann_ready_for_search(self, vectors_by_fragment_id: dict[int, list[float]]) -> None:
        status = self.ann_index.inspect(
            fragment_ids=vectors_by_fragment_id.keys(),
            embedding_model=self.config.embedding_model,
        )
        if status.get("ready"):
            return
        with self._ann_maintenance.acquire_exclusive(timeout_ms=self.tuning.ann_wait_timeout_ms) as acquired:
            if not acquired:
                raise SearchStatusError(self._ann_maintenance.read_status(current_mode="lite"))
            status = self.ann_index.inspect(
                fragment_ids=vectors_by_fragment_id.keys(),
                embedding_model=self.config.embedding_model,
            )
            if status.get("ready"):
                return
            try:
                with self._ann_maintenance.active_status(
                    code="rebuild_in_progress",
                    phase="ann_rebuild",
                    fragment_count=len(vectors_by_fragment_id),
                    current_mode="lite",
                    suggested_action="retry",
                ):
                    self.ann_index.ensure_ready(
                        vectors_by_fragment_id=vectors_by_fragment_id,
                        embedding_model=self.config.embedding_model,
                    )
            except Exception:
                raise SearchStatusError(self._ann_unavailable_status(reason="rebuild_failed"))
            status = self.ann_index.inspect(
                fragment_ids=vectors_by_fragment_id.keys(),
                embedding_model=self.config.embedding_model,
            )
            if not status.get("ready"):
                raise SearchStatusError(self._ann_unavailable_status(reason="rebuild_failed"))

    def _ann_unavailable_status(self, *, reason: str) -> dict:
        return {
            "code": "ann_unavailable",
            "retryable": False,
            "reason": reason,
            "current_mode": "lite",
            "suggested_action": "fallback_or_surface_error",
        }

    def _status_response(self, status: dict) -> SearchResponse:
        return SearchResponse(items=[], count=0, status=status)

    def _select_compression_candidate(self, protected_fragment_ids: set[int]) -> Optional[Fragment]:
        fragments = [
            fragment
            for fragment in self.store.list_fragments()
            if fragment.layer == "working" and fragment.id not in protected_fragment_ids
        ]
        if not fragments:
            return None
        deviations = self._deviations(fragments)
        pressure = self._pressure_working()
        best: Optional[tuple[float, float, int, int, Fragment]] = None
        for fragment in fragments:
            deviation = deviations[fragment.id]
            failure_penalty = self.tuning.failure_penalty_base ** fragment.compression_fail_count
            score = pressure * abs(deviation) / failure_penalty
            candidate = (score, abs(deviation), fragment.content_length, -fragment.id, fragment)
            if best is None or candidate > best:
                best = candidate
        return None if best is None else best[-1]

    def _select_delete_candidate(self, protected_fragment_ids: set[int]) -> Optional[Fragment]:
        fragments = [
            fragment
            for fragment in self.store.list_fragments()
            if fragment.layer == "holding" and fragment.id not in protected_fragment_ids
        ]
        if not fragments:
            return None
        return min(
            fragments,
            key=lambda fragment: (self._search_priority(fragment.id), fragment.id),
        )

    def _deviations(self, fragments: list[Fragment]) -> dict[int, float]:
        if len(fragments) == 1:
            return {fragments[0].id: 0.0}
        values = {fragment.id: math.log1p(self._search_priority(fragment.id)) for fragment in fragments}
        center = median(values.values())
        spread = median(abs(value - center) for value in values.values()) + self.tuning.epsilon
        return {fragment_id: (value - center) / spread for fragment_id, value in values.items()}

    def _reference_score(self, fragment_id: int) -> float:
        return math.log1p(self._reference_signal(fragment_id))

    def _reference_signal(self, fragment_id: int) -> float:
        current_anchor_id = self.store.max_anchor_sequence()
        current_anchor_span = max(1, current_anchor_id)
        signal = 0.0
        for reference in self.store.list_references_for_fragment(fragment_id):
            distance = max(0, current_anchor_id - reference.from_anchor_id)
            signal += math.exp(-distance / math.sqrt(current_anchor_span))
        return signal

    def _confidence_score(self, fragment_id: int) -> float:
        feedback = self.store.list_feedback_for_fragment(fragment_id)
        if not feedback:
            return 0.5
        values = [self._feedback_value(item.verdict) for item in feedback]
        feedback_mean = sum(values) / len(values)
        return (feedback_mean + 1.0) / 2.0

    def _search_priority(self, fragment_id: int) -> float:
        return self._reference_score(fragment_id) * self._confidence_score(fragment_id)

    def _feedback_value(self, verdict: str) -> int:
        mapping = {"positive": 1, "neutral": 0, "negative": -1}
        return mapping[verdict]

    def _effective_working_ratio(self) -> float:
        ratio = self.tuning.initial_working_ratio
        lower = self.tuning.working_ratio_floor
        upper = 1.0 - lower
        span = upper - lower
        if span <= 0:
            return ratio
        metrics = self.store.list_sequence_metrics()
        for index in range(len(metrics)):
            window = metrics[max(0, index - self.tuning.rate_window_size + 1): index + 1]
            compress_rate = sum(item.compress_count for item in window) / self.tuning.rate_window_size
            delete_rate = sum(item.delete_count for item in window) / self.tuning.rate_window_size
            delta = compress_rate - delete_rate
            up_room = (upper - ratio) / span
            down_room = (ratio - lower) / span
            if delta >= 0:
                ratio = ratio + (up_room * delta / self.tuning.rate_window_size)
            else:
                ratio = ratio + (down_room * delta / self.tuning.rate_window_size)
            ratio = clamp(ratio, lower, upper)
        return ratio

    def _working_budget(self) -> int:
        return self._budget_snapshot()["working_budget"]

    def _holding_budget(self) -> int:
        return self._budget_snapshot()["holding_budget"]

    def _pressure_working(self) -> float:
        budget = self._working_budget()
        if budget <= 0:
            return 0.0
        return self._current_layer_usage("working") / budget

    def _record_sequence_metrics(self, *, anchor_id: int, counters: dict[str, int]) -> None:
        usage = self._layer_usage_snapshot()
        self.store.record_sequence_metrics(
            anchor_id=anchor_id,
            working_usage_bytes=usage["working"],
            holding_usage_bytes=usage["holding"],
            compress_count=counters["compress"],
            delete_count=counters["delete"],
        )

    def _layer_usage_exceeds_budget(self, layer: str, *, extra_bytes: int = 0) -> bool:
        usage = self._current_layer_usage(layer) + extra_bytes
        budgets = self._budget_snapshot()
        budget_key = f"{layer}_budget"
        return usage > budgets[budget_key]

    def _budget_snapshot(self) -> dict[str, float | int]:
        working_ratio = self._effective_working_ratio()
        total_capacity = self._total_capacity_bytes()
        working_budget = int(total_capacity * working_ratio)
        return {
            "working_ratio": working_ratio,
            "working_budget": working_budget,
            "holding_budget": total_capacity - working_budget,
        }

    def _layer_usage_snapshot(self) -> dict[str, int]:
        return {
            "working": self._current_layer_usage("working"),
            "holding": self._current_layer_usage("holding"),
        }

    def _current_layer_usage(self, layer: str) -> int:
        return sum(
            fragment.content_length
            for fragment in self.store.list_fragments()
            if fragment.layer == layer
        )

    def _estimate_token_count(self, content: str) -> int:
        return max(1, math.ceil(self._content_size(content) / 4))

    def _sort_fragments_by_search_priority(self, fragments: Iterable[Fragment]) -> list[Fragment]:
        ordered = list(fragments)
        ordered.sort(key=lambda fragment: (-self._search_priority(fragment.id), fragment.id))
        return ordered

    def _recent_sequence_metrics(self) -> list:
        metrics = self.store.list_sequence_metrics()
        return metrics[-self.tuning.rate_window_size :]

    def _total_capacity_bytes(self) -> int:
        return int(self.config.total_capacity_mb * (1 << 20))

    def _build_fragment_view(self, fragment: Fragment) -> FragmentView:
        anchor = self._require_anchor(fragment.anchor_id)
        return FragmentView(
            id=fragment.id,
            anchor_id=fragment.anchor_id,
            reply_to=anchor.replies_to_anchor_id,
            kind=fragment.kind,
            content=fragment.content,
            content_length=fragment.content_length,
            layer=fragment.layer,
            compression_fail_count=fragment.compression_fail_count,
            reference_score=self._reference_score(fragment.id),
            confidence_score=self._confidence_score(fragment.id),
            search_priority=self._search_priority(fragment.id),
        )

    def _build_search_item_view(
        self,
        fragment: Fragment,
        *,
        diagnostics: Optional[dict] = None,
    ) -> SearchItemView:
        anchor = self._require_anchor(fragment.anchor_id)
        return SearchItemView(
            id=fragment.id,
            anchor_id=fragment.anchor_id,
            parent_id=fragment.parent_id,
            actor=fragment.actor,
            reply_to=anchor.replies_to_anchor_id,
            kind=fragment.kind,
            content=fragment.content,
            content_length=fragment.content_length,
            layer=fragment.layer,
            reference_score=self._reference_score(fragment.id),
            confidence_score=self._confidence_score(fragment.id),
            search_priority=self._search_priority(fragment.id),
            diagnostics=diagnostics,
        )

    def _lexical_diagnostics(
        self,
        lexical_rank: tuple[int, int, int, float],
        retrieval_mode: str,
    ) -> dict:
        all_terms_present, exact_match, matched_terms, coverage = lexical_rank
        return {
            "retrieval_mode": retrieval_mode,
            "lexical_rank": {
                "all_terms_present": bool(all_terms_present),
                "exact_match": bool(exact_match),
                "matched_term_count": matched_terms,
                "coverage": coverage,
            },
        }

    def _require_anchor(self, anchor_id: int) -> Anchor:
        anchor = self.store.get_anchor(anchor_id)
        if anchor is None:
            raise ValueError("anchor not found")
        return anchor

    def _content_size(self, content: str) -> int:
        return self.store.content_size(content)
