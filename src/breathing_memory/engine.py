from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import re
from statistics import median
from typing import Iterable, Optional

from .compression import CodexExecCompressionBackend, CompressionBackend
from .config import EngineTuning, MemoryConfig
from .models import Anchor, Fragment, FragmentFeedback, FragmentReference
from .store import SQLiteStore


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


LEXICAL_SEPARATOR_PATTERN = re.compile(r"[`'\"/\\._\-,:;!?()[\]{}<>|*+=~@#$%^&]+|[、。・「」『』【】（）！？：；　]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
SEARCH_TERM_PATTERN = re.compile(r"[0-9a-z]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+")


class BreathingMemoryEngine:
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        tuning: Optional[EngineTuning] = None,
        store: Optional[SQLiteStore] = None,
        compression_backend: Optional[CompressionBackend] = None,
    ):
        self.config = config or MemoryConfig()
        self.tuning = tuning or EngineTuning()
        self._validate_retrieval_mode(self.config.retrieval_mode)
        self.store = store or SQLiteStore(Path(self.config.db_path))
        self.compression_backend = compression_backend or CodexExecCompressionBackend()

    def close(self) -> None:
        self.store.close()

    def remember(
        self,
        content: str,
        actor: str,
        reply_to: Optional[int] = None,
        source_fragment_ids: Optional[Iterable[int]] = None,
    ) -> dict:
        if actor not in {"user", "agent"}:
            raise ValueError("actor must be 'user' or 'agent'")
        if not content or not content.strip():
            raise ValueError("content must not be empty")
        if reply_to is not None and not self.store.anchor_exists(reply_to):
            raise ValueError("reply_to anchor not found")

        normalized_sources = self._normalize_source_fragment_ids(source_fragment_ids)
        self._validate_source_fragment_ids(normalized_sources)

        duplicate_fragment = self._find_duplicate_agent_fragment(
            actor=actor,
            content=content,
            reply_to=reply_to,
        )
        if duplicate_fragment is not None:
            self._merge_material_references(
                from_anchor_id=duplicate_fragment.anchor_id,
                source_fragment_ids=normalized_sources,
            )
            self._apply_observation_driven_promotion(normalized_sources)
            return self._serialize_fragment(duplicate_fragment)

        anchor_id = self.store.create_anchor(
            replies_to_anchor_id=reply_to,
            is_root=reply_to is None,
        )
        fragment_id = self.store.create_fragment(
            anchor_id=anchor_id,
            parent_id=None,
            actor=actor,
            content=content,
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
        self.store.record_sequence_metrics(
            anchor_id=anchor_id,
            working_usage_bytes=self._current_layer_usage("working"),
            holding_usage_bytes=self._current_layer_usage("holding"),
            compress_count=counters["compress"],
            delete_count=counters["delete"],
        )
        fragment = self.store.get_fragment(fragment_id)
        if fragment is None:
            raise RuntimeError("remembered fragment was not preserved")
        return self._serialize_fragment(fragment)

    def search(
        self,
        query: str,
        result_count: Optional[int] = None,
        search_effort: Optional[int] = None,
    ) -> dict:
        normalized_result_count = self._normalize_result_count(result_count)
        self._normalize_search_effort(search_effort)
        retrieval_mode = self._resolve_retrieval_mode()
        if retrieval_mode in {"lite", "default"}:
            raise RuntimeError(f"retrieval mode '{retrieval_mode}' is not supported in this text-only slice")

        normalized_query = self._normalize_lexical_text(query)
        query_terms = self._search_terms(normalized_query)
        fragments = self.store.list_fragments()
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
            self._serialize_search_item(fragment)
            for fragment, _ in lexical_matches[:normalized_result_count]
        ]
        return {"items": items, "count": len(items)}

    def fetch(
        self,
        fragment_id: Optional[int] = None,
        anchor_id: Optional[int] = None,
    ) -> dict:
        if (fragment_id is None) == (anchor_id is None):
            raise ValueError("exactly one of fragment_id or anchor_id is required")
        if fragment_id is not None:
            fragment = self.store.get_fragment(fragment_id)
            if fragment is None:
                return {"items": [], "count": 0}
            return {"items": [self._serialize_search_item(fragment)], "count": 1}

        assert anchor_id is not None
        fragments = self.store.list_fragments_by_anchor(anchor_id)
        fragments.sort(key=lambda fragment: (-self._search_priority(fragment.id), fragment.id))
        return {
            "items": [self._serialize_search_item(fragment) for fragment in fragments],
            "count": len(fragments),
        }

    def feedback(
        self,
        from_anchor_id: int,
        fragment_id: int,
        verdict: str,
    ) -> dict:
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
        return {
            "fragment_id": fragment_id,
            "verdict": verdict,
            "confidence_score": self._confidence_score(fragment_id),
            "search_priority": self._search_priority(fragment_id),
        }

    def stats(self) -> dict:
        fragments = self.store.list_fragments()
        working_ratio = self._effective_working_ratio()
        recent_metrics = self._recent_sequence_metrics()
        parameters = asdict(self.tuning)
        parameters["retrieval_mode"] = self.config.retrieval_mode
        parameters["total_capacity_mb"] = self.config.total_capacity_mb
        parameters["total_capacity"] = self._total_capacity_bytes()
        return {
            "fragment_count": len(fragments),
            "working_count": len([fragment for fragment in fragments if fragment.layer == "working"]),
            "holding_count": len([fragment for fragment in fragments if fragment.layer == "holding"]),
            "working_usage": self._current_layer_usage("working"),
            "holding_usage": self._current_layer_usage("holding"),
            "working_budget": self._working_budget(),
            "holding_budget": self._holding_budget(),
            "working_ratio": working_ratio,
            "recent_compress_count": sum(metric.compress_count for metric in recent_metrics),
            "recent_delete_count": sum(metric.delete_count for metric in recent_metrics),
            "parameters": parameters,
        }

    def export_fragment_graph(self) -> list[dict]:
        return [self._serialize_search_item(fragment) for fragment in self.store.list_fragments()]

    def _validate_retrieval_mode(self, retrieval_mode: str) -> None:
        if retrieval_mode not in {"auto", "super_lite", "lite", "default"}:
            raise ValueError("retrieval_mode must be 'auto', 'super_lite', 'lite', or 'default'")

    def _resolve_retrieval_mode(self) -> str:
        if self.config.retrieval_mode == "auto":
            return "super_lite"
        return self.config.retrieval_mode

    def _normalize_result_count(self, result_count: Optional[int]) -> int:
        value = self.tuning.default_result_count if result_count is None else int(result_count)
        if value < self.tuning.default_result_count or value % self.tuning.default_result_count != 0:
            raise ValueError("result_count must be 8 * 2^n")
        quotient = value // self.tuning.default_result_count
        if quotient & (quotient - 1):
            raise ValueError("result_count must be 8 * 2^n")
        return value

    def _normalize_search_effort(self, search_effort: Optional[int]) -> int:
        value = self.tuning.default_search_effort if search_effort is None else int(search_effort)
        if value < self.tuning.default_search_effort or value % self.tuning.default_search_effort != 0:
            raise ValueError("search_effort must be 32 * 2^n")
        quotient = value // self.tuning.default_search_effort
        if quotient & (quotient - 1):
            raise ValueError("search_effort must be 32 * 2^n")
        return value

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

    def _find_duplicate_agent_fragment(
        self,
        actor: str,
        content: str,
        reply_to: Optional[int],
    ) -> Optional[Fragment]:
        if actor != "agent" or reply_to is None:
            return None

        normalized_content = self._normalize_dedup_content(content)
        for candidate in self.store.list_root_fragments_replying_to_anchor(reply_to, actor="agent"):
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
            if self._current_layer_usage("working") > self._working_budget():
                if not self._compress_once(counters, protected_fragment_ids):
                    raise RuntimeError("unable to free working capacity")
                continue
            if self._current_layer_usage("holding") > self._holding_budget():
                if not self._delete_once(counters, protected_fragment_ids):
                    raise RuntimeError("unable to free holding capacity")
                continue
            return

    def _compress_once(self, counters: dict[str, int], protected_fragment_ids: set[int]) -> bool:
        candidate = self._select_compression_candidate(protected_fragment_ids)
        if candidate is None:
            return False
        result = self.compression_backend.compress(
            candidate.content,
            self.tuning.compression_ratio,
        )
        if not result.content.strip() or self._content_size(result.content) >= candidate.content_length:
            self.store.increment_compression_fail_count(candidate.id)
            return False

        while self._current_layer_usage("holding") + candidate.content_length > self._holding_budget():
            if not self._delete_once(counters, protected_fragment_ids=set()):
                self.store.increment_compression_fail_count(candidate.id)
                return False

        child_id = self.store.create_fragment(
            anchor_id=candidate.anchor_id,
            parent_id=candidate.id,
            actor=candidate.actor,
            content=result.content,
            layer="working",
        )
        self.store.copy_references(candidate.id, child_id)
        self.store.copy_feedback(candidate.id, child_id)
        self.store.create_reference(from_anchor_id=candidate.anchor_id, fragment_id=child_id)
        self.store.update_fragment_layer(candidate.id, "holding")
        counters["compress"] += 1
        return True

    def _delete_once(self, counters: dict[str, int], protected_fragment_ids: set[int]) -> bool:
        candidate = self._select_delete_candidate(protected_fragment_ids)
        if candidate is None:
            return False
        self.store.delete_fragment(candidate.id)
        counters["delete"] += 1
        return True

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
        return int(self._total_capacity_bytes() * self._effective_working_ratio())

    def _holding_budget(self) -> int:
        return self._total_capacity_bytes() - self._working_budget()

    def _pressure_working(self) -> float:
        budget = self._working_budget()
        if budget <= 0:
            return 0.0
        return self._current_layer_usage("working") / budget

    def _current_layer_usage(self, layer: str) -> int:
        return sum(
            fragment.content_length
            for fragment in self.store.list_fragments()
            if fragment.layer == layer
        )

    def _recent_sequence_metrics(self) -> list:
        metrics = self.store.list_sequence_metrics()
        return metrics[-self.tuning.rate_window_size :]

    def _total_capacity_bytes(self) -> int:
        return int(self.config.total_capacity_mb * (1 << 20))

    def _serialize_fragment(self, fragment: Fragment) -> dict:
        anchor = self._require_anchor(fragment.anchor_id)
        return {
            "id": fragment.id,
            "anchor_id": fragment.anchor_id,
            "reply_to": anchor.replies_to_anchor_id,
            "content": fragment.content,
            "content_length": fragment.content_length,
            "layer": fragment.layer,
            "compression_fail_count": fragment.compression_fail_count,
            "reference_score": self._reference_score(fragment.id),
            "confidence_score": self._confidence_score(fragment.id),
            "search_priority": self._search_priority(fragment.id),
        }

    def _serialize_search_item(self, fragment: Fragment) -> dict:
        anchor = self._require_anchor(fragment.anchor_id)
        return {
            "id": fragment.id,
            "anchor_id": fragment.anchor_id,
            "parent_id": fragment.parent_id,
            "actor": fragment.actor,
            "reply_to": anchor.replies_to_anchor_id,
            "content": fragment.content,
            "content_length": fragment.content_length,
            "layer": fragment.layer,
            "reference_score": self._reference_score(fragment.id),
            "confidence_score": self._confidence_score(fragment.id),
            "search_priority": self._search_priority(fragment.id),
        }

    def _require_anchor(self, anchor_id: int) -> Anchor:
        anchor = self.store.get_anchor(anchor_id)
        if anchor is None:
            raise ValueError("anchor not found")
        return anchor

    def _content_size(self, content: str) -> int:
        return self.store.content_size(content)
