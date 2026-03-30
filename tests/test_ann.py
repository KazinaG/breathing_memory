from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from breathing_memory.ann import HnswIndex


class _FakeNumpy:
    @staticmethod
    def asarray(value, dtype=None):
        del dtype
        return value


class _FakeLoadedIndex:
    def __init__(self, labels: list[int]):
        self.labels = labels
        self.ef_values: list[int] = []
        self.k_values: list[int] = []

    def set_ef(self, value: int) -> None:
        self.ef_values.append(int(value))

    def knn_query(self, vectors, k: int):
        del vectors
        self.k_values.append(int(k))
        return [self.labels[:k]], [[0.0] * min(k, len(self.labels))]


class _FakePersistentIndex:
    storage: dict[str, dict] = {}

    def __init__(self, *, space: str, dim: int):
        self.space = space
        self.dim = dim
        self.max_elements = 0
        self.ef_value: int | None = None
        self.items: dict[int, list[float]] = {}
        self.deleted: set[int] = set()
        self.resized_to: list[int] = []

    def init_index(self, *, max_elements: int, ef_construction: int, M: int) -> None:
        del ef_construction, M
        self.max_elements = int(max_elements)

    def set_ef(self, value: int) -> None:
        self.ef_value = int(value)

    def add_items(self, vectors, labels) -> None:
        for label, vector in zip(labels, vectors):
            self.items[int(label)] = list(map(float, vector))

    def save_index(self, path: str) -> None:
        _FakePersistentIndex.storage[path] = {
            "dim": self.dim,
            "max_elements": self.max_elements,
            "items": {label: list(vector) for label, vector in self.items.items()},
            "deleted": sorted(self.deleted),
        }
        Path(path).write_text("fake-index", encoding="utf-8")

    def load_index(self, path: str, max_elements: int) -> None:
        payload = _FakePersistentIndex.storage[path]
        self.max_elements = int(max_elements)
        self.items = {int(label): list(vector) for label, vector in payload["items"].items()}
        self.deleted = {int(label) for label in payload["deleted"]}

    def mark_deleted(self, label: int) -> None:
        self.deleted.add(int(label))

    def resize_index(self, capacity: int) -> None:
        self.max_elements = int(capacity)
        self.resized_to.append(int(capacity))

    def knn_query(self, vectors, k: int):
        del vectors
        labels = [label for label in sorted(self.items) if label not in self.deleted]
        return [labels[:k]], [[0.0] * min(k, len(labels))]


class _FakeHnswLib:
    def Index(self, *, space: str, dim: int):
        return _FakePersistentIndex(space=space, dim=dim)


class _TestableHnswIndex(HnswIndex):
    def support_available(self) -> bool:
        return True

    def _import_numpy(self):
        return _FakeNumpy()

    def _import_hnswlib(self):
        return _FakeHnswLib()


class HnswIndexTests(unittest.TestCase):
    def test_query_uses_search_effort_without_fixed_floor(self) -> None:
        index = _TestableHnswIndex(Path("/tmp/query-floor.sqlite3"))
        index._index = _FakeLoadedIndex(labels=list(range(1, 17)))
        index._active_fragment_ids = set(range(1, 17))

        result = index.query(vector=[1.0, 0.0], limit=8, search_effort=32)

        self.assertEqual(result, list(range(1, 9)))
        self.assertEqual(index._index.ef_values, [32])
        self.assertEqual(index._index.k_values, [8])

    def test_query_raises_ef_to_limit_when_limit_exceeds_search_effort(self) -> None:
        index = _TestableHnswIndex(Path("/tmp/query-limit.sqlite3"))
        index._index = _FakeLoadedIndex(labels=list(range(1, 65)))
        index._active_fragment_ids = set(range(1, 65))

        result = index.query(vector=[1.0, 0.0], limit=64, search_effort=32)

        self.assertEqual(result, list(range(1, 65)))
        self.assertEqual(index._index.ef_values, [64])
        self.assertEqual(index._index.k_values, [64])

    def test_inspect_reports_missing_index_files_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            index = _TestableHnswIndex(Path(tempdir) / "memory.sqlite3")

            status = index.inspect(fragment_ids=[1, 2], embedding_model="model-a")

            self.assertFalse(status["ready"])
            self.assertEqual(status["status"], "build_required")
            self.assertEqual(status["reason"], "missing_index_files")

    def test_rebuild_persists_metadata_and_makes_index_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            index = _TestableHnswIndex(Path(tempdir) / "memory.sqlite3")

            index.rebuild(
                vectors_by_fragment_id={
                    2: [0.0, 1.0],
                    1: [1.0, 0.0],
                },
                embedding_model="model-a",
            )

            status = index.inspect(fragment_ids=[1, 2], embedding_model="model-a")
            metadata = index._read_metadata()

            self.assertTrue(status["ready"])
            self.assertEqual(status["reason"], "healthy_index")
            self.assertIsNotNone(metadata)
            assert metadata is not None
            self.assertEqual(metadata.embedding_model, "model-a")
            self.assertEqual(metadata.dimension, 2)
            self.assertEqual(metadata.fragment_ids, [1, 2])

    def test_remove_persists_fragment_set_without_deleted_label(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            index = _TestableHnswIndex(Path(tempdir) / "memory.sqlite3")
            index.rebuild(
                vectors_by_fragment_id={
                    1: [1.0, 0.0],
                    2: [0.0, 1.0],
                },
                embedding_model="model-a",
            )

            index.remove(1)

            metadata = index._read_metadata()
            status = index.inspect(fragment_ids=[2], embedding_model="model-a")

            self.assertIsNotNone(metadata)
            assert metadata is not None
            self.assertEqual(metadata.fragment_ids, [2])
            self.assertTrue(status["ready"])
            self.assertEqual(status["reason"], "healthy_index")

    def test_append_resizes_capacity_when_new_fragment_exceeds_current_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            index = _TestableHnswIndex(Path(tempdir) / "memory.sqlite3")
            index.rebuild(vectors_by_fragment_id={1: [1.0, 0.0]}, embedding_model="model-a")
            index._capacity = 1

            index.append(fragment_id=2, vector=[0.0, 1.0], embedding_model="model-a")

            metadata = index._read_metadata()
            self.assertIsNotNone(metadata)
            assert metadata is not None
            self.assertEqual(metadata.fragment_ids, [1, 2])
            self.assertGreaterEqual(metadata.capacity, 2)


if __name__ == "__main__":
    unittest.main()
