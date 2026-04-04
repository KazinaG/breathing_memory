from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RememberRequest:
    content: str
    actor: str
    reply_to: int | None = None
    source_fragment_ids: tuple[int, ...] = ()
    kind: str | None = None


@dataclass(frozen=True, slots=True)
class RecentRequest:
    limit: int = 4
    actor: str | None = None
    reply_to: int | None = None


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    result_count: int | None = None
    search_effort: int | None = None
    actor: str | None = None
    kind: str | None = None
    include_diagnostics: bool = False


@dataclass(frozen=True, slots=True)
class ReadActiveCollaborationPolicyRequest:
    token_budget: int | None = None


@dataclass(frozen=True, slots=True)
class FetchRequest:
    fragment_id: int | None = None
    anchor_id: int | None = None


@dataclass(frozen=True, slots=True)
class FeedbackRequest:
    from_anchor_id: int
    fragment_id: int
    verdict: str


@dataclass(frozen=True, slots=True)
class FragmentView:
    id: int
    anchor_id: int
    reply_to: int | None
    kind: str | None
    content: str
    content_length: int
    layer: str
    compression_fail_count: int
    reference_score: float
    confidence_score: float
    search_priority: float

@dataclass(frozen=True, slots=True)
class SearchItemView:
    id: int
    anchor_id: int
    parent_id: int | None
    actor: str
    reply_to: int | None
    kind: str | None
    content: str
    content_length: int
    layer: str
    reference_score: float
    confidence_score: float
    search_priority: float
    diagnostics: dict[str, Any] | None = None

@dataclass(frozen=True, slots=True)
class SearchResponse:
    items: list[SearchItemView] = field(default_factory=list)
    count: int = 0
    status: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ReadActiveCollaborationPolicyResponse:
    items: list[SearchItemView] = field(default_factory=list)
    count: int = 0
    token_budget: int = 0
    used_token_budget: int = 0
    truncated: bool = False



@dataclass(frozen=True, slots=True)
class FeedbackResult:
    fragment_id: int
    verdict: str
    confidence_score: float
    search_priority: float

@dataclass(frozen=True, slots=True)
class StatsResult:
    fragment_count: int
    working_count: int
    holding_count: int
    working_usage: int
    holding_usage: int
    working_budget: int
    holding_budget: int
    working_ratio: float
    recent_compress_count: int
    recent_delete_count: int
    parameters: dict[str, Any]
