from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Anchor:
    id: int
    replies_to_anchor_id: Optional[int]
    is_root: bool


@dataclass(frozen=True)
class Fragment:
    id: int
    anchor_id: int
    parent_id: Optional[int]
    actor: str
    content: str
    content_length: int
    embedding_vector: Optional[bytes]
    layer: str
    compression_fail_count: int


@dataclass(frozen=True)
class FragmentReference:
    id: int
    from_anchor_id: int
    fragment_id: int


@dataclass(frozen=True)
class FragmentFeedback:
    id: int
    from_anchor_id: int
    fragment_id: int
    verdict: str


@dataclass(frozen=True)
class SequenceMetric:
    anchor_id: int
    working_usage_bytes: int
    holding_usage_bytes: int
    compress_count: int
    delete_count: int
