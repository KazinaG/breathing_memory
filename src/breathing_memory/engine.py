from __future__ import annotations

from typing import Iterable, Optional

from .adapters.serializers import (
    feedback_result_to_payload,
    fragment_view_to_payload,
    read_active_collaboration_policy_response_to_payload,
    search_response_to_payload,
    stats_result_to_payload,
)
from .core.service import BreathingMemoryEngine as CoreBreathingMemoryEngine
from .core.service import SearchStatusError
from .core.types import (
    FeedbackRequest,
    FeedbackResult,
    FetchRequest,
    FragmentView,
    ReadActiveCollaborationPolicyRequest,
    ReadActiveCollaborationPolicyResponse,
    RecentRequest,
    RememberRequest,
    SearchRequest,
    SearchResponse,
    StatsResult,
)


def _remember_request(
    content: str,
    actor: str,
    *,
    reply_to: Optional[int],
    source_fragment_ids: Optional[Iterable[int]],
    kind: Optional[str],
) -> RememberRequest:
    return RememberRequest(
        content=content,
        actor=actor,
        reply_to=reply_to,
        source_fragment_ids=tuple(source_fragment_ids or ()),
        kind=kind,
    )


def _recent_request(limit: int, actor: Optional[str], reply_to: Optional[int]) -> RecentRequest:
    return RecentRequest(limit=limit, actor=actor, reply_to=reply_to)


def _search_request(
    query: str,
    result_count: Optional[int],
    search_effort: Optional[int],
    actor: Optional[str],
    kind: Optional[str],
    include_diagnostics: bool,
) -> SearchRequest:
    return SearchRequest(
        query=query,
        result_count=result_count,
        search_effort=search_effort,
        actor=actor,
        kind=kind,
        include_diagnostics=include_diagnostics,
    )


def _policy_request(
    token_budget: Optional[int],
) -> ReadActiveCollaborationPolicyRequest:
    return ReadActiveCollaborationPolicyRequest(token_budget=token_budget)


def _fetch_request(fragment_id: Optional[int], anchor_id: Optional[int]) -> FetchRequest:
    return FetchRequest(fragment_id=fragment_id, anchor_id=anchor_id)


def _feedback_request(from_anchor_id: int, fragment_id: int, verdict: str) -> FeedbackRequest:
    return FeedbackRequest(
        from_anchor_id=from_anchor_id,
        fragment_id=fragment_id,
        verdict=verdict,
    )


class BreathingMemoryEngine(CoreBreathingMemoryEngine):
    """Legacy compatibility shim over the typed core service.

    New in-process consumers should prefer `breathing_memory.core.BreathingMemoryEngine`
    and the typed request/response API. This shim exists to preserve the older
    dict-shaped method surface and import path.
    """

    def _remember_payload(self, request: RememberRequest) -> dict:
        return fragment_view_to_payload(super().remember(request))

    def _recent_payload(self, request: RecentRequest) -> dict:
        return search_response_to_payload(super().recent(request))

    def _search_payload(self, request: SearchRequest) -> dict:
        return search_response_to_payload(super().search(request))

    def _policy_payload(self, request: ReadActiveCollaborationPolicyRequest) -> dict:
        return read_active_collaboration_policy_response_to_payload(
            super().read_active_collaboration_policy(request)
        )

    def _fetch_payload(self, request: FetchRequest) -> dict:
        return search_response_to_payload(super().fetch(request))

    def _feedback_payload(self, request: FeedbackRequest) -> dict:
        return feedback_result_to_payload(super().feedback(request))

    def remember_typed(self, request: RememberRequest) -> FragmentView:
        return super().remember(request)

    def remember(
        self,
        content: str | RememberRequest,
        actor: str | None = None,
        reply_to: Optional[int] = None,
        source_fragment_ids: Optional[Iterable[int]] = None,
        kind: Optional[str] = None,
    ) -> dict:
        if isinstance(content, RememberRequest):
            return self._remember_payload(content)
        if actor is None:
            raise TypeError("BreathingMemoryEngine.remember() missing 1 required positional argument: 'actor'")
        return self._remember_payload(
            _remember_request(
                content,
                actor,
                reply_to=reply_to,
                source_fragment_ids=source_fragment_ids,
                kind=kind,
            )
        )

    def recent_typed(self, request: RecentRequest) -> SearchResponse:
        return super().recent(request)

    def recent(
        self,
        limit: int | RecentRequest = 4,
        actor: Optional[str] = None,
        reply_to: Optional[int] = None,
    ) -> dict:
        if isinstance(limit, RecentRequest):
            return self._recent_payload(limit)
        return self._recent_payload(_recent_request(limit, actor=actor, reply_to=reply_to))

    def search_typed(self, request: SearchRequest) -> SearchResponse:
        return super().search(request)

    def search(
        self,
        query: str | SearchRequest,
        result_count: Optional[int] = None,
        search_effort: Optional[int] = None,
        actor: Optional[str] = None,
        kind: Optional[str] = None,
        include_diagnostics: bool = False,
    ) -> dict:
        if isinstance(query, SearchRequest):
            return self._search_payload(query)
        return self._search_payload(
            _search_request(
                query,
                result_count=result_count,
                search_effort=search_effort,
                actor=actor,
                kind=kind,
                include_diagnostics=include_diagnostics,
            )
        )

    def read_active_collaboration_policy_typed(
        self,
        request: ReadActiveCollaborationPolicyRequest,
    ) -> ReadActiveCollaborationPolicyResponse:
        return super().read_active_collaboration_policy(request)

    def read_active_collaboration_policy(
        self,
        token_budget: Optional[int] | ReadActiveCollaborationPolicyRequest = None,
    ) -> dict:
        if isinstance(token_budget, ReadActiveCollaborationPolicyRequest):
            return self._policy_payload(token_budget)
        return self._policy_payload(_policy_request(token_budget))

    def fetch_typed(self, request: FetchRequest) -> SearchResponse:
        return super().fetch(request)

    def fetch(
        self,
        fragment_id: Optional[int] | FetchRequest = None,
        anchor_id: Optional[int] = None,
    ) -> dict:
        if isinstance(fragment_id, FetchRequest):
            return self._fetch_payload(fragment_id)
        return self._fetch_payload(_fetch_request(fragment_id=fragment_id, anchor_id=anchor_id))

    def feedback_typed(self, request: FeedbackRequest) -> FeedbackResult:
        return super().feedback(request)

    def feedback(
        self,
        from_anchor_id: int | FeedbackRequest,
        fragment_id: int | None = None,
        verdict: str | None = None,
    ) -> dict:
        if isinstance(from_anchor_id, FeedbackRequest):
            return self._feedback_payload(from_anchor_id)
        if fragment_id is None or verdict is None:
            raise TypeError(
                "BreathingMemoryEngine.feedback() missing required positional arguments: 'fragment_id' and 'verdict'"
            )
        return self._feedback_payload(_feedback_request(from_anchor_id, fragment_id, verdict))

    def stats_typed(self) -> StatsResult:
        return super().stats()

    # Dict-shaped compatibility methods used by the historic import path and MCP shim.

    def stats(self) -> dict:
        return stats_result_to_payload(super().stats())


__all__ = [
    "BreathingMemoryEngine",
    "SearchStatusError",
]
