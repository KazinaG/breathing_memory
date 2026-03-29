from __future__ import annotations

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


class _TestableHnswIndex(HnswIndex):
    def support_available(self) -> bool:
        return True

    def _import_numpy(self):
        return _FakeNumpy()


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


if __name__ == "__main__":
    unittest.main()
