from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from ..compression import CompressionResult
from ..models import Anchor, Fragment, FragmentFeedback, FragmentReference, SequenceMetric


class Store(Protocol):
    db_path: Path

    def close(self) -> None:
        ...

    def anchor_exists(self, anchor_id: int) -> bool:
        ...

    def fragment_exists(self, fragment_id: int) -> bool:
        ...

    def create_anchor(self, replies_to_anchor_id: Optional[int], is_root: bool) -> int:
        ...

    def create_fragment(
        self,
        anchor_id: int,
        actor: str,
        content: str,
        layer: str,
        kind: Optional[str] = None,
        parent_id: Optional[int] = None,
        embedding_vector: Optional[bytes] = None,
    ) -> int:
        ...

    def update_fragment_embedding(self, fragment_id: int, embedding_vector: Optional[bytes]) -> None:
        ...

    def update_fragment_layer(self, fragment_id: int, layer: str) -> None:
        ...

    def increment_compression_fail_count(self, fragment_id: int) -> None:
        ...

    def create_reference(self, from_anchor_id: int, fragment_id: int) -> int:
        ...

    def create_feedback(self, from_anchor_id: int, fragment_id: int, verdict: str) -> int:
        ...

    def copy_references(self, source_fragment_id: int, target_fragment_id: int) -> None:
        ...

    def copy_feedback(self, source_fragment_id: int, target_fragment_id: int) -> None:
        ...

    def delete_fragment(self, fragment_id: int) -> None:
        ...

    def get_anchor(self, anchor_id: int) -> Anchor | None:
        ...

    def get_fragment(self, fragment_id: int) -> Fragment | None:
        ...

    def list_fragments(self, actor: Optional[str] = None, kind: Optional[str] = None) -> list[Fragment]:
        ...

    def list_fragments_by_anchor(self, anchor_id: int) -> list[Fragment]:
        ...

    def list_recent_root_fragments(
        self,
        limit: int,
        actor: Optional[str] = None,
        reply_to_anchor_id: Optional[int] = None,
    ) -> list[Fragment]:
        ...

    def list_root_fragments_replying_to_anchor(
        self,
        anchor_id: int,
        *,
        actor: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> list[Fragment]:
        ...

    def list_references_for_fragment(self, fragment_id: int) -> list[FragmentReference]:
        ...

    def list_references_from_anchor(self, anchor_id: int) -> list[FragmentReference]:
        ...

    def list_feedback_for_fragment(self, fragment_id: int) -> list[FragmentFeedback]:
        ...

    def list_sequence_metrics(self) -> list[SequenceMetric]:
        ...

    def record_sequence_metrics(
        self,
        *,
        anchor_id: int,
        working_usage_bytes: int,
        holding_usage_bytes: int,
        compress_count: int,
        delete_count: int,
    ) -> None:
        ...

    def max_anchor_sequence(self) -> int:
        ...

    def content_size(self, content: str) -> int:
        ...


class EmbeddingBackend(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def warmup(self) -> None:
        ...

    def start_background_warmup(
        self,
        *,
        on_error: Callable[[Exception], None] | None = None,
    ) -> bool:
        ...


class AnnIndex(Protocol):
    def support_available(self) -> bool:
        ...

    def inspect(self, *, fragment_ids: Sequence[int], embedding_model: str) -> dict[str, Any]:
        ...

    def ensure_ready(
        self,
        *,
        vectors_by_fragment_id: Mapping[int, Sequence[float]],
        embedding_model: str,
    ) -> dict[str, Any]:
        ...

    def rebuild(
        self,
        *,
        vectors_by_fragment_id: Mapping[int, Sequence[float]],
        embedding_model: str,
    ) -> None:
        ...

    def append(self, *, fragment_id: int, vector: Sequence[float], embedding_model: str) -> None:
        ...

    def remove(self, fragment_id: int) -> None:
        ...

    def query(self, *, vector: Sequence[float], limit: int, search_effort: int) -> list[int]:
        ...


class CompressionBackend(Protocol):
    def compress(self, content: str, compression_ratio: float) -> CompressionResult:
        ...
