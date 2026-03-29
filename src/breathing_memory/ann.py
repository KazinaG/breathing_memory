from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


INDEX_FORMAT_VERSION = 1
DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_CONSTRUCTION = 128
DEFAULT_HNSW_EF_SEARCH = 64
MIN_HNSW_CAPACITY = 16


class ApproximateNearestNeighborIndex(Protocol):
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


@dataclass(frozen=True)
class HnswIndexMetadata:
    version: int
    embedding_model: str
    dimension: int
    capacity: int
    fragment_ids: list[int]


def hnswlib_available() -> bool:
    try:
        import hnswlib  # noqa: F401
    except Exception:
        return False
    return True


class HnswIndex:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.index_path = self.db_path.with_suffix(f"{self.db_path.suffix}.hnsw.bin")
        self.metadata_path = self.db_path.with_suffix(f"{self.db_path.suffix}.hnsw.json")
        self._index = None
        self._embedding_model: str | None = None
        self._dimension: int | None = None
        self._capacity = 0
        self._active_fragment_ids: set[int] = set()

    def support_available(self) -> bool:
        return hnswlib_available()

    def inspect(self, *, fragment_ids: Sequence[int], embedding_model: str) -> dict[str, Any]:
        fragment_id_list = sorted({int(fragment_id) for fragment_id in fragment_ids})
        return self._inspect(fragment_id_list=fragment_id_list, embedding_model=embedding_model)

    def ensure_ready(
        self,
        *,
        vectors_by_fragment_id: Mapping[int, Sequence[float]],
        embedding_model: str,
    ) -> dict[str, Any]:
        fragment_id_list = sorted(int(fragment_id) for fragment_id in vectors_by_fragment_id)
        status = self._inspect(fragment_id_list=fragment_id_list, embedding_model=embedding_model)
        if not self.support_available():
            raise RuntimeError("retrieval mode 'default' requires HNSW index support")
        if not fragment_id_list:
            self._clear_loaded_index()
            return status
        if status["ready"]:
            metadata = self._read_metadata()
            if metadata is None:
                raise RuntimeError("HNSW index metadata disappeared during readiness check")
            try:
                self._load_index(metadata)
            except Exception:
                self._rebuild(vectors_by_fragment_id=vectors_by_fragment_id, embedding_model=embedding_model)
                return {
                    "support_available": True,
                    "ready": True,
                    "status": "ready",
                    "reason": "rebuilt_invalid_index_binary",
                    "index_path": str(self.index_path),
                    "metadata_path": str(self.metadata_path),
                    "fragment_count": len(fragment_id_list),
                }
            return status
        self._rebuild(vectors_by_fragment_id=vectors_by_fragment_id, embedding_model=embedding_model)
        return {
            "support_available": True,
            "ready": True,
            "status": "ready",
            "reason": f"rebuilt_{status['reason']}",
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "fragment_count": len(fragment_id_list),
        }

    def rebuild(
        self,
        *,
        vectors_by_fragment_id: Mapping[int, Sequence[float]],
        embedding_model: str,
    ) -> None:
        self._rebuild(vectors_by_fragment_id=vectors_by_fragment_id, embedding_model=embedding_model)

    def append(self, *, fragment_id: int, vector: Sequence[float], embedding_model: str) -> None:
        if not self.support_available():
            raise RuntimeError("retrieval mode 'default' requires HNSW index support")
        metadata = self._read_metadata()
        if metadata is None:
            raise RuntimeError("HNSW index is not ready")
        if metadata.embedding_model != embedding_model:
            raise RuntimeError("HNSW index metadata does not match the active embedding model")
        if metadata.dimension != len(vector):
            raise RuntimeError("HNSW index dimension does not match the embedding vector")
        self._load_index(metadata)
        if fragment_id in self._active_fragment_ids:
            return
        if len(self._active_fragment_ids) + 1 > self._capacity:
            self._resize_capacity(max(len(self._active_fragment_ids) + 1, self._capacity * 2))
        numpy = self._import_numpy()
        self._index.add_items(
            numpy.asarray([list(map(float, vector))], dtype="float32"),
            numpy.asarray([int(fragment_id)], dtype="int64"),
        )
        self._active_fragment_ids.add(int(fragment_id))
        self._persist_loaded_index()

    def remove(self, fragment_id: int) -> None:
        if not self.support_available():
            return
        metadata = self._read_metadata()
        if metadata is None or int(fragment_id) not in metadata.fragment_ids:
            return
        self._load_index(metadata)
        if int(fragment_id) not in self._active_fragment_ids:
            return
        self._index.mark_deleted(int(fragment_id))
        self._active_fragment_ids.remove(int(fragment_id))
        self._persist_loaded_index()

    def query(self, *, vector: Sequence[float], limit: int, search_effort: int) -> list[int]:
        if not self.support_available():
            raise RuntimeError("retrieval mode 'default' requires HNSW index support")
        if self._index is None or not self._active_fragment_ids:
            return []
        search_limit = min(int(limit), len(self._active_fragment_ids))
        if search_limit <= 0:
            return []
        self._index.set_ef(max(int(search_effort), search_limit))
        numpy = self._import_numpy()
        labels, _ = self._index.knn_query(
            numpy.asarray([list(map(float, vector))], dtype="float32"),
            k=search_limit,
        )
        return [int(label) for label in labels[0] if int(label) >= 0]

    def _inspect(self, *, fragment_id_list: list[int], embedding_model: str) -> dict[str, Any]:
        if not self.support_available():
            return {
                "support_available": False,
                "ready": False,
                "status": "unavailable",
                "reason": "hnsw_support_unavailable",
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
                "fragment_count": len(fragment_id_list),
            }
        if not fragment_id_list:
            return {
                "support_available": True,
                "ready": False,
                "status": "build_required",
                "reason": "empty_corpus",
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
                "fragment_count": 0,
            }
        if not self.index_path.exists() or not self.metadata_path.exists():
            return {
                "support_available": True,
                "ready": False,
                "status": "build_required",
                "reason": "missing_index_files",
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
                "fragment_count": len(fragment_id_list),
            }
        metadata = self._read_metadata()
        if metadata is None:
            return {
                "support_available": True,
                "ready": False,
                "status": "rebuild_required",
                "reason": "invalid_metadata",
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
                "fragment_count": len(fragment_id_list),
            }
        if metadata.version != INDEX_FORMAT_VERSION:
            return {
                "support_available": True,
                "ready": False,
                "status": "rebuild_required",
                "reason": "index_version_mismatch",
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
                "fragment_count": len(fragment_id_list),
            }
        if metadata.embedding_model != embedding_model:
            return {
                "support_available": True,
                "ready": False,
                "status": "rebuild_required",
                "reason": "embedding_model_mismatch",
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
                "fragment_count": len(fragment_id_list),
            }
        if metadata.fragment_ids != fragment_id_list:
            return {
                "support_available": True,
                "ready": False,
                "status": "rebuild_required",
                "reason": "fragment_set_mismatch",
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
                "fragment_count": len(fragment_id_list),
            }
        return {
            "support_available": True,
            "ready": True,
            "status": "ready",
            "reason": "healthy_index",
            "index_path": str(self.index_path),
            "metadata_path": str(self.metadata_path),
            "fragment_count": len(fragment_id_list),
            "dimension": metadata.dimension,
        }

    def _rebuild(
        self,
        *,
        vectors_by_fragment_id: Mapping[int, Sequence[float]],
        embedding_model: str,
    ) -> None:
        if not self.support_available():
            raise RuntimeError("retrieval mode 'default' requires HNSW index support")
        if not vectors_by_fragment_id:
            self._clear_loaded_index()
            return
        items = sorted((int(fragment_id), list(map(float, vector))) for fragment_id, vector in vectors_by_fragment_id.items())
        dimension = len(items[0][1])
        if dimension <= 0:
            raise RuntimeError("cannot build an HNSW index with zero-dimensional embeddings")
        for _, vector in items:
            if len(vector) != dimension:
                raise RuntimeError("HNSW rebuild requires embeddings with a consistent dimension")

        hnswlib = self._import_hnswlib()
        numpy = self._import_numpy()
        capacity = max(MIN_HNSW_CAPACITY, len(items))
        index = hnswlib.Index(space="cosine", dim=dimension)
        index.init_index(
            max_elements=capacity,
            ef_construction=DEFAULT_HNSW_EF_CONSTRUCTION,
            M=DEFAULT_HNSW_M,
        )
        index.set_ef(DEFAULT_HNSW_EF_SEARCH)
        index.add_items(
            numpy.asarray([vector for _, vector in items], dtype="float32"),
            numpy.asarray([fragment_id for fragment_id, _ in items], dtype="int64"),
        )
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        index.save_index(str(self.index_path))
        metadata = HnswIndexMetadata(
            version=INDEX_FORMAT_VERSION,
            embedding_model=embedding_model,
            dimension=dimension,
            capacity=capacity,
            fragment_ids=[fragment_id for fragment_id, _ in items],
        )
        self.metadata_path.write_text(
            json.dumps(
                {
                    "version": metadata.version,
                    "embedding_model": metadata.embedding_model,
                    "dimension": metadata.dimension,
                    "capacity": metadata.capacity,
                    "fragment_ids": metadata.fragment_ids,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self._index = index
        self._embedding_model = embedding_model
        self._dimension = dimension
        self._capacity = capacity
        self._active_fragment_ids = set(metadata.fragment_ids)

    def _load_index(self, metadata: HnswIndexMetadata) -> None:
        if (
            self._index is not None
            and self._embedding_model == metadata.embedding_model
            and self._dimension == metadata.dimension
            and self._active_fragment_ids == set(metadata.fragment_ids)
        ):
            return
        hnswlib = self._import_hnswlib()
        index = hnswlib.Index(space="cosine", dim=metadata.dimension)
        index.load_index(str(self.index_path), max_elements=max(metadata.capacity, len(metadata.fragment_ids), MIN_HNSW_CAPACITY))
        index.set_ef(DEFAULT_HNSW_EF_SEARCH)
        self._index = index
        self._embedding_model = metadata.embedding_model
        self._dimension = metadata.dimension
        self._capacity = max(metadata.capacity, len(metadata.fragment_ids), MIN_HNSW_CAPACITY)
        self._active_fragment_ids = set(metadata.fragment_ids)

    def _persist_loaded_index(self) -> None:
        if self._index is None or self._embedding_model is None or self._dimension is None:
            raise RuntimeError("cannot persist an unloaded HNSW index")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(self.index_path))
        metadata = {
            "version": INDEX_FORMAT_VERSION,
            "embedding_model": self._embedding_model,
            "dimension": self._dimension,
            "capacity": max(self._capacity, len(self._active_fragment_ids), MIN_HNSW_CAPACITY),
            "fragment_ids": sorted(self._active_fragment_ids),
        }
        self.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def _resize_capacity(self, capacity: int) -> None:
        if self._index is None:
            raise RuntimeError("cannot resize an unloaded HNSW index")
        self._index.resize_index(max(int(capacity), MIN_HNSW_CAPACITY))
        self._capacity = max(int(capacity), MIN_HNSW_CAPACITY)

    def _clear_loaded_index(self) -> None:
        self._index = None
        self._embedding_model = None
        self._dimension = None
        self._capacity = 0
        self._active_fragment_ids = set()

    def _read_metadata(self) -> HnswIndexMetadata | None:
        if not self.metadata_path.exists():
            return None
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        try:
            fragment_ids = [int(value) for value in payload["fragment_ids"]]
            return HnswIndexMetadata(
                version=int(payload["version"]),
                embedding_model=str(payload["embedding_model"]),
                dimension=int(payload["dimension"]),
                capacity=int(payload.get("capacity", max(len(fragment_ids), MIN_HNSW_CAPACITY))),
                fragment_ids=sorted(fragment_ids),
            )
        except Exception:
            return None

    def _import_hnswlib(self):
        import hnswlib

        return hnswlib

    def _import_numpy(self):
        import numpy

        return numpy
