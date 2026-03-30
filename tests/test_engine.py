from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest

from breathing_memory.compression import CompressionResult, StubCompressionBackend
from breathing_memory.config import EngineTuning, MemoryConfig
from breathing_memory.embeddings import StubEmbeddingBackend, cosine_similarity
from breathing_memory.engine import BreathingMemoryEngine
from breathing_memory.store import SQLiteStore


class StubAnnIndex:
    def __init__(self, support: bool = True):
        self.support = support
        self.embedding_model: str | None = None
        self.vectors_by_fragment_id: dict[int, list[float]] = {}
        self.force_reason: str | None = None
        self.raise_on_rebuild: Exception | None = None
        self.rebuild_calls = 0

    def support_available(self) -> bool:
        return self.support

    def inspect(self, *, fragment_ids, embedding_model):
        fragment_id_list = sorted(set(int(fragment_id) for fragment_id in fragment_ids))
        if not self.support:
            return self._status(False, "unavailable", "hnsw_support_unavailable", fragment_id_list)
        if self.force_reason is not None:
            return self._status(False, "rebuild_required", self.force_reason, fragment_id_list)
        if not fragment_id_list:
            return self._status(False, "build_required", "empty_corpus", fragment_id_list)
        if self.embedding_model != embedding_model:
            if not self.vectors_by_fragment_id:
                return self._status(False, "build_required", "missing_index_files", fragment_id_list)
            return self._status(False, "rebuild_required", "embedding_model_mismatch", fragment_id_list)
        if sorted(self.vectors_by_fragment_id) != fragment_id_list:
            reason = "missing_index_files" if not self.vectors_by_fragment_id else "fragment_set_mismatch"
            status = "build_required" if not self.vectors_by_fragment_id else "rebuild_required"
            return self._status(False, status, reason, fragment_id_list)
        return self._status(True, "ready", "healthy_index", fragment_id_list)

    def ensure_ready(self, *, vectors_by_fragment_id, embedding_model):
        status = self.inspect(fragment_ids=vectors_by_fragment_id.keys(), embedding_model=embedding_model)
        if status["ready"]:
            return status
        self.rebuild(vectors_by_fragment_id=vectors_by_fragment_id, embedding_model=embedding_model)
        return self.inspect(fragment_ids=vectors_by_fragment_id.keys(), embedding_model=embedding_model)

    def rebuild(self, *, vectors_by_fragment_id, embedding_model):
        self.rebuild_calls += 1
        if self.raise_on_rebuild is not None:
            raise self.raise_on_rebuild
        self.embedding_model = embedding_model
        self.vectors_by_fragment_id = {
            int(fragment_id): list(map(float, vector))
            for fragment_id, vector in vectors_by_fragment_id.items()
        }
        self.force_reason = None

    def append(self, *, fragment_id, vector, embedding_model):
        if not self.support:
            raise RuntimeError("HNSW support unavailable")
        self.embedding_model = embedding_model
        self.vectors_by_fragment_id[int(fragment_id)] = list(map(float, vector))

    def remove(self, fragment_id):
        self.vectors_by_fragment_id.pop(int(fragment_id), None)

    def query(self, *, vector, limit, search_effort):
        del search_effort
        candidates = sorted(
            self.vectors_by_fragment_id.items(),
            key=lambda item: (-((cosine_similarity(vector, item[1]) + 1.0) / 2.0), item[0]),
        )
        return [fragment_id for fragment_id, _ in candidates[:limit]]

    def _status(self, ready: bool, status: str, reason: str, fragment_ids: list[int]) -> dict:
        return {
            "support_available": self.support,
            "ready": ready,
            "status": status,
            "reason": reason,
            "index_path": "/tmp/stub.hnsw.bin",
            "metadata_path": "/tmp/stub.hnsw.json",
            "fragment_count": len(fragment_ids),
        }


class SelectiveCompressionBackend:
    def __init__(self, results_by_content: dict[str, CompressionResult]):
        self.results_by_content = results_by_content

    def compress(self, content: str, compression_ratio: float) -> CompressionResult:
        del compression_ratio
        return self.results_by_content[content]


def make_engine(
    root: Path,
    total_capacity: int = 160,
    retrieval_mode: str = "super_lite",
    embedding_backend: StubEmbeddingBackend | None = None,
    ann_index: StubAnnIndex | None = None,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    tuning: EngineTuning | None = None,
) -> BreathingMemoryEngine:
    config = MemoryConfig(
        db_path=root / "memory.sqlite3",
        total_capacity_mb=total_capacity / (1024 * 1024),
        retrieval_mode=retrieval_mode,
        embedding_model=embedding_model,
    )
    return BreathingMemoryEngine(
        config=config,
        tuning=tuning,
        compression_backend=StubCompressionBackend(),
        embedding_backend=embedding_backend,
        ann_index=ann_index,
    )


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.engine = make_engine(self.root)

    def tearDown(self) -> None:
        self.engine.close()
        self.tempdir.cleanup()

    def test_remember_creates_anchor_fragment_relations_and_metrics(self) -> None:
        fragment = self.engine.remember(content="alpha memory", actor="user")

        anchors = self.engine.store.list_anchors()
        fragments = self.engine.store.list_fragments()
        references = self.engine.store.list_references()
        feedback = self.engine.store.list_feedback()
        metrics = self.engine.store.list_sequence_metrics()

        self.assertEqual(len(anchors), 1)
        self.assertEqual(len(fragments), 1)
        self.assertEqual(fragment["anchor_id"], anchors[0].id)
        self.assertIsNone(fragment["reply_to"])
        self.assertEqual(references[0].from_anchor_id, anchors[0].id)
        self.assertEqual(references[0].fragment_id, fragment["id"])
        self.assertEqual(feedback[0].verdict, "positive")
        self.assertEqual(metrics[0].anchor_id, anchors[0].id)
        self.assertEqual(metrics[0].compress_count, 0)
        self.assertEqual(metrics[0].delete_count, 0)
        self.assertIsNone(fragment["kind"])

    def test_remember_rejects_unknown_reply_to_anchor(self) -> None:
        with self.assertRaisesRegex(ValueError, "reply_to anchor not found"):
            self.engine.remember(content="bad reply", actor="agent", reply_to=9999)

    def test_feedback_uses_verdicts_and_updates_confidence(self) -> None:
        fragment = self.engine.remember(content="beta memory", actor="user")

        self.engine.feedback(
            from_anchor_id=fragment["anchor_id"],
            fragment_id=fragment["id"],
            verdict="negative",
        )

        self.assertEqual(self.engine._confidence_score(fragment["id"]), 0.5)

    def test_legacy_database_is_reset_and_backed_up(self) -> None:
        db_path = self.root / "legacy.sqlite3"
        connection = sqlite3.connect(str(db_path))
        try:
            connection.executescript(
                """
                CREATE TABLE memory_fragments (
                    id TEXT PRIMARY KEY,
                    node_id TEXT NOT NULL,
                    reply_to TEXT NULL,
                    content TEXT NOT NULL,
                    content_length INTEGER NOT NULL,
                    created_turn INTEGER NOT NULL,
                    layer TEXT NOT NULL,
                    initial_confidence REAL NOT NULL,
                    compression_fail_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE reference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fragment_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    actor TEXT NOT NULL,
                    weight REAL NOT NULL,
                    source_fragment_ids_json TEXT NULL
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

        store = SQLiteStore(db_path)
        try:
            self.assertEqual(store.list_fragments(), [])
            backups = list(self.root.glob("legacy.legacy-*.sqlite3"))
            self.assertEqual(len(backups), 1)
        finally:
            store.close()

    def test_existing_database_without_kind_column_is_migrated(self) -> None:
        db_path = self.root / "missing-kind.sqlite3"
        connection = sqlite3.connect(str(db_path))
        try:
            connection.executescript(
                """
                CREATE TABLE anchors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    replies_to_anchor_id INTEGER NULL REFERENCES anchors(id) ON DELETE SET NULL,
                    is_root INTEGER NOT NULL CHECK (is_root IN (0, 1))
                );

                CREATE TABLE fragments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    anchor_id INTEGER NOT NULL REFERENCES anchors(id) ON DELETE CASCADE,
                    parent_id INTEGER NULL REFERENCES fragments(id) ON DELETE SET NULL,
                    actor TEXT NOT NULL CHECK (actor IN ('user', 'agent')),
                    content TEXT NOT NULL,
                    content_length INTEGER NOT NULL,
                    embedding_vector BLOB NULL,
                    layer TEXT NOT NULL CHECK (layer IN ('working', 'holding')),
                    compression_fail_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE fragment_references (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_anchor_id INTEGER NOT NULL REFERENCES anchors(id) ON DELETE CASCADE,
                    fragment_id INTEGER NOT NULL REFERENCES fragments(id) ON DELETE CASCADE
                );

                CREATE TABLE fragment_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_anchor_id INTEGER NOT NULL REFERENCES anchors(id) ON DELETE CASCADE,
                    fragment_id INTEGER NOT NULL REFERENCES fragments(id) ON DELETE CASCADE,
                    verdict TEXT NOT NULL CHECK (verdict IN ('positive', 'neutral', 'negative'))
                );

                CREATE TABLE sequence_metrics (
                    anchor_id INTEGER PRIMARY KEY REFERENCES anchors(id) ON DELETE CASCADE,
                    working_usage_bytes INTEGER NOT NULL,
                    holding_usage_bytes INTEGER NOT NULL,
                    compress_count INTEGER NOT NULL,
                    delete_count INTEGER NOT NULL
                );
                """
            )
            connection.commit()
        finally:
            connection.close()

        store = SQLiteStore(db_path)
        try:
            columns = store.connection.execute("PRAGMA table_info(fragments)").fetchall()
            column_names = {row["name"] for row in columns}
            self.assertIn("kind", column_names)
            indexes = store.connection.execute("PRAGMA index_list(fragments)").fetchall()
            index_names = {row["name"] for row in indexes}
            self.assertIn("idx_fragments_kind_id", index_names)
        finally:
            store.close()

    def test_search_does_not_record_references(self) -> None:
        fragment = self.engine.remember(content="search me", actor="user")
        before = len(self.engine.store.list_references())

        result = self.engine.search("search", result_count=8, search_effort=32)

        self.assertEqual(result["items"][0]["id"], fragment["id"])
        self.assertEqual(len(self.engine.store.list_references()), before)
        self.assertNotIn("diagnostics", result["items"][0])

    def test_remember_persists_kind_and_search_can_filter_by_kind(self) -> None:
        policy = self.engine.remember(
            content="Prefer concise answers first.",
            actor="agent",
            kind="collaboration_policy",
        )
        self.engine.remember(content="Regular memory", actor="agent")

        result = self.engine.search(
            "concise answers",
            result_count=8,
            search_effort=32,
            kind="collaboration_policy",
        )

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["id"], policy["id"])
        self.assertEqual(result["items"][0]["kind"], "collaboration_policy")

    def test_search_can_filter_by_actor(self) -> None:
        agent = self.engine.remember(content="agent-authored summary", actor="agent")
        self.engine.remember(content="user-authored summary", actor="user")

        result = self.engine.search(
            "summary",
            result_count=8,
            search_effort=32,
            actor="agent",
        )

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["id"], agent["id"])
        self.assertEqual(result["items"][0]["actor"], "agent")

    def test_search_rejects_unknown_actor_filter(self) -> None:
        self.engine.remember(content="search me", actor="user")

        with self.assertRaisesRegex(ValueError, "actor must be 'user' or 'agent'"):
            self.engine.search("search", actor="system")

    def test_search_validates_result_count_and_search_effort(self) -> None:
        self.engine.remember(content="search me", actor="user")

        with self.assertRaisesRegex(ValueError, "result_count must be 8 \\* 2\\^n"):
            self.engine.search("search", result_count=12)
        with self.assertRaisesRegex(ValueError, "search_effort must be 32 \\* 2\\^n"):
            self.engine.search("search", search_effort=48)

    def test_search_accepts_power_of_two_result_count_and_search_effort(self) -> None:
        fragment = self.engine.remember(content="search me", actor="user")

        result = self.engine.search("search", result_count=16, search_effort=64)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["id"], fragment["id"])

    def test_search_rejects_lite_without_embedding_backend(self) -> None:
        self.engine.close()
        self.engine = make_engine(self.root / "lite", retrieval_mode="lite")
        self.engine.embedding_backend = None
        self.engine.remember(content="search me", actor="user")

        with self.assertRaisesRegex(RuntimeError, "requires an embedding backend"):
            self.engine.search("search")

    def test_search_rejects_default_without_hnsw_support(self) -> None:
        backend = StubEmbeddingBackend({"search me": [1.0, 0.0], "search": [1.0, 0.0]})
        self.engine.close()
        self.engine = make_engine(
            self.root / "default-unavailable",
            retrieval_mode="default",
            embedding_backend=backend,
            ann_index=StubAnnIndex(support=False),
        )
        self.engine.remember(content="search me", actor="user")

        with self.assertRaisesRegex(RuntimeError, "HNSW index support"):
            self.engine.search("search")

    def test_default_search_returns_results_when_hnsw_is_available(self) -> None:
        backend = StubEmbeddingBackend(
            {
                "default winner": [0.8, 0.2],
                "default query": [0.8, 0.2],
            }
        )
        self.engine.close()
        self.engine = make_engine(
            self.root / "default-ready",
            retrieval_mode="default",
            embedding_backend=backend,
            ann_index=StubAnnIndex(),
        )
        fragment = self.engine.remember(content="default winner", actor="user")

        result = self.engine.search("default query", result_count=8, search_effort=32)

        self.assertEqual(result["items"][0]["id"], fragment["id"])

    def test_lite_search_reranks_semantic_candidates_by_search_priority(self) -> None:
        backend = StubEmbeddingBackend(
            {
                "high priority lexical mismatch": [1.0, 0.0],
                "semantic winner": [0.8, 0.2],
                "semantic query": [0.8, 0.2],
            }
        )
        self.engine.close()
        self.engine = make_engine(self.root / "semantic", retrieval_mode="lite", embedding_backend=backend)
        high = self.engine.remember(content="high priority lexical mismatch", actor="user")
        semantic = self.engine.remember(content="semantic winner", actor="user")
        self.engine.feedback(from_anchor_id=semantic["anchor_id"], fragment_id=semantic["id"], verdict="positive")
        self.engine.feedback(from_anchor_id=semantic["anchor_id"], fragment_id=semantic["id"], verdict="positive")

        result = self.engine.search("semantic query", result_count=8, search_effort=32)

        self.assertEqual(result["items"][0]["id"], semantic["id"])
        self.assertEqual(result["items"][1]["id"], high["id"])

    def test_lite_search_can_include_semantic_diagnostics(self) -> None:
        backend = StubEmbeddingBackend(
            {
                "semantic winner": [0.8, 0.2],
                "semantic query": [0.8, 0.2],
            }
        )
        self.engine.close()
        self.engine = make_engine(self.root / "semantic-diag", retrieval_mode="lite", embedding_backend=backend)
        fragment = self.engine.remember(content="semantic winner", actor="user")

        result = self.engine.search(
            "semantic query",
            result_count=8,
            search_effort=32,
            include_diagnostics=True,
        )

        self.assertEqual(result["items"][0]["id"], fragment["id"])
        diagnostics = result["items"][0]["diagnostics"]
        self.assertEqual(diagnostics["retrieval_mode"], "lite")
        self.assertIn("semantic_similarity", diagnostics)
        self.assertIn("normalized_priority", diagnostics)
        self.assertIn("ranking_score", diagnostics)

    def test_default_search_can_include_semantic_diagnostics(self) -> None:
        backend = StubEmbeddingBackend(
            {
                "default winner": [0.8, 0.2],
                "default query": [0.8, 0.2],
            }
        )
        self.engine.close()
        self.engine = make_engine(
            self.root / "default-diag",
            retrieval_mode="default",
            embedding_backend=backend,
            ann_index=StubAnnIndex(),
        )
        fragment = self.engine.remember(content="default winner", actor="user")

        result = self.engine.search(
            "default query",
            result_count=8,
            search_effort=32,
            include_diagnostics=True,
        )

        self.assertEqual(result["items"][0]["id"], fragment["id"])
        diagnostics = result["items"][0]["diagnostics"]
        self.assertEqual(diagnostics["retrieval_mode"], "default")
        self.assertIn("semantic_similarity", diagnostics)
        self.assertIn("normalized_priority", diagnostics)
        self.assertIn("ranking_score", diagnostics)

    def test_lexical_search_can_include_diagnostics(self) -> None:
        fragment = self.engine.remember(content="memory\nsearch ready", actor="user")

        result = self.engine.search(
            "search memory",
            result_count=8,
            search_effort=32,
            include_diagnostics=True,
        )

        self.assertEqual(result["items"][0]["id"], fragment["id"])
        diagnostics = result["items"][0]["diagnostics"]
        self.assertEqual(diagnostics["retrieval_mode"], "super_lite")
        self.assertTrue(diagnostics["lexical_rank"]["all_terms_present"])
        self.assertEqual(diagnostics["lexical_rank"]["matched_term_count"], 2)

    def test_auto_mode_uses_default_when_hnsw_is_ready(self) -> None:
        backend = StubEmbeddingBackend({"alpha semantic": [1.0, 0.0], "alpha": [1.0, 0.0]})
        self.engine.close()
        self.engine = make_engine(
            self.root / "auto-default",
            retrieval_mode="auto",
            embedding_backend=backend,
            ann_index=StubAnnIndex(),
        )
        fragment = self.engine.remember(content="alpha semantic", actor="user")

        result = self.engine.search("alpha", result_count=8, search_effort=32)

        self.assertEqual(result["items"][0]["id"], fragment["id"])
        self.assertEqual(self.engine._resolve_retrieval_mode(), "default")

    def test_auto_mode_falls_back_to_lite_when_hnsw_is_unavailable(self) -> None:
        backend = StubEmbeddingBackend({"alpha semantic": [1.0, 0.0], "alpha": [1.0, 0.0]})
        self.engine.close()
        self.engine = make_engine(
            self.root / "auto-lite",
            retrieval_mode="auto",
            embedding_backend=backend,
            ann_index=StubAnnIndex(support=False),
        )
        fragment = self.engine.remember(content="alpha semantic", actor="user")

        result = self.engine.search("alpha", result_count=8, search_effort=32, include_diagnostics=True)

        self.assertEqual(result["items"][0]["id"], fragment["id"])
        self.assertEqual(self.engine._resolve_retrieval_mode(), "lite")
        self.assertEqual(result["items"][0]["diagnostics"]["retrieval_mode"], "lite")

    def test_auto_mode_repairs_hnsw_when_supported_but_not_ready(self) -> None:
        backend = StubEmbeddingBackend({"alpha semantic": [1.0, 0.0], "alpha": [1.0, 0.0]})
        ann_index = StubAnnIndex()
        self.engine.close()
        self.engine = make_engine(
            self.root / "auto-repair",
            retrieval_mode="auto",
            embedding_backend=backend,
            ann_index=ann_index,
        )
        fragment = self.engine.remember(content="alpha semantic", actor="user")
        ann_index.force_reason = "fragment_set_mismatch"

        result = self.engine.search("alpha", result_count=8, search_effort=32, include_diagnostics=True)

        self.assertEqual(result["items"][0]["id"], fragment["id"])
        self.assertEqual(result["items"][0]["diagnostics"]["retrieval_mode"], "default")
        self.assertGreaterEqual(ann_index.rebuild_calls, 2)

    def test_auto_mode_returns_retryable_status_when_ann_rebuild_is_in_progress(self) -> None:
        backend = StubEmbeddingBackend({"alpha semantic": [1.0, 0.0], "alpha": [1.0, 0.0]})
        ann_index = StubAnnIndex()
        self.engine.close()
        self.engine = make_engine(
            self.root / "auto-rebuild-pending",
            retrieval_mode="auto",
            embedding_backend=backend,
            ann_index=ann_index,
            tuning=EngineTuning(ann_wait_timeout_ms=10),
        )
        self.engine.remember(content="alpha semantic", actor="user")
        search_engine = make_engine(
            self.root / "auto-rebuild-pending",
            retrieval_mode="auto",
            embedding_backend=backend,
            ann_index=ann_index,
            tuning=EngineTuning(ann_wait_timeout_ms=10),
        )

        try:
            with self.engine._ann_maintenance.acquire_exclusive():
                result = search_engine.search("alpha", result_count=8, search_effort=32)
        finally:
            search_engine.close()

        self.assertEqual(result["items"], [])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["status"]["code"], "rebuild_in_progress")
        self.assertTrue(result["status"]["retryable"])
        self.assertEqual(result["status"]["current_mode"], "lite")

    def test_auto_mode_returns_non_retryable_status_when_ann_rebuild_fails(self) -> None:
        backend = StubEmbeddingBackend({"alpha semantic": [1.0, 0.0], "alpha": [1.0, 0.0]})
        ann_index = StubAnnIndex()
        self.engine.close()
        self.engine = make_engine(
            self.root / "auto-rebuild-fails",
            retrieval_mode="auto",
            embedding_backend=backend,
            ann_index=ann_index,
            tuning=EngineTuning(ann_wait_timeout_ms=10),
        )
        self.engine.remember(content="alpha semantic", actor="user")
        ann_index.force_reason = "fragment_set_mismatch"
        ann_index.raise_on_rebuild = RuntimeError("boom")

        result = self.engine.search("alpha", result_count=8, search_effort=32)

        self.assertEqual(result["items"], [])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["status"]["code"], "ann_unavailable")
        self.assertFalse(result["status"]["retryable"])
        self.assertEqual(result["status"]["reason"], "rebuild_failed")

    def test_lite_search_backfills_missing_embeddings(self) -> None:
        backend = StubEmbeddingBackend({"needs embedding": [1.0, 0.0], "needs": [1.0, 0.0]})
        self.engine.close()
        self.engine = make_engine(self.root / "backfill", retrieval_mode="lite", embedding_backend=backend)
        fragment = self.engine.remember(content="needs embedding", actor="user")
        self.engine.store.update_fragment_embedding(fragment["id"], None)

        result = self.engine.search("needs", result_count=8, search_effort=32)

        stored = self.engine.store.get_fragment(fragment["id"])
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertIsNotNone(stored.embedding_vector)
        self.assertEqual(result["items"][0]["id"], fragment["id"])

    def test_default_search_backfills_missing_embeddings_and_rebuilds_index(self) -> None:
        backend = StubEmbeddingBackend({"needs embedding": [1.0, 0.0], "needs": [1.0, 0.0]})
        ann_index = StubAnnIndex()
        self.engine.close()
        self.engine = make_engine(
            self.root / "default-backfill",
            retrieval_mode="default",
            embedding_backend=backend,
            ann_index=ann_index,
        )
        fragment = self.engine.remember(content="needs embedding", actor="user")
        self.engine.store.update_fragment_embedding(fragment["id"], None)
        ann_index.vectors_by_fragment_id = {}

        result = self.engine.search("needs", result_count=8, search_effort=32)

        stored = self.engine.store.get_fragment(fragment["id"])
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertIsNotNone(stored.embedding_vector)
        self.assertEqual(result["items"][0]["id"], fragment["id"])
        self.assertIn(fragment["id"], ann_index.vectors_by_fragment_id)

    def test_default_search_rebuilds_when_index_is_marked_invalid(self) -> None:
        backend = StubEmbeddingBackend(
            {
                "semantic winner": [0.8, 0.2],
                "semantic query": [0.8, 0.2],
            }
        )
        ann_index = StubAnnIndex()
        self.engine.close()
        self.engine = make_engine(
            self.root / "default-rebuild",
            retrieval_mode="default",
            embedding_backend=backend,
            ann_index=ann_index,
        )
        fragment = self.engine.remember(content="semantic winner", actor="user")
        ann_index.force_reason = "invalid_metadata"

        result = self.engine.search("semantic query", result_count=8, search_effort=32)

        self.assertEqual(result["items"][0]["id"], fragment["id"])
        self.assertIsNone(ann_index.force_reason)

    def test_default_search_rebuilds_when_embedding_model_changes(self) -> None:
        backend = StubEmbeddingBackend(
            {
                "semantic winner": [0.8, 0.2],
                "semantic query": [0.8, 0.2],
            }
        )
        ann_index = StubAnnIndex()
        self.engine.close()
        self.engine = make_engine(
            self.root / "default-model-a",
            retrieval_mode="default",
            embedding_backend=backend,
            ann_index=ann_index,
            embedding_model="model-a",
        )
        fragment = self.engine.remember(content="semantic winner", actor="user")

        self.engine.close()
        self.engine = make_engine(
            self.root / "default-model-a",
            retrieval_mode="default",
            embedding_backend=backend,
            ann_index=ann_index,
            embedding_model="model-b",
        )

        result = self.engine.search("semantic query", result_count=8, search_effort=32)

        self.assertEqual(result["items"][0]["id"], fragment["id"])
        self.assertEqual(ann_index.embedding_model, "model-b")

    def test_search_normalizes_symbols_and_whitespace(self) -> None:
        fragment = self.engine.remember(content="`public/` 側には\nまだ残っています。", actor="user")

        result = self.engine.search("public 側にはまだ残っています", result_count=8, search_effort=32)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["id"], fragment["id"])

    def test_search_matches_query_terms_without_exact_contiguous_substring(self) -> None:
        fragment = self.engine.remember(content="memory\nsearch ready", actor="user")

        result = self.engine.search("search memory", result_count=8, search_effort=32)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["id"], fragment["id"])

    def test_observation_driven_promotion_moves_holding_fragment_back_to_working(self) -> None:
        source = self.engine.remember(content="x" * 40, actor="user")
        self.engine.store.update_fragment_layer(source["id"], "holding")

        source_fragment = self.engine.store.get_fragment(source["id"])
        self.assertIsNotNone(source_fragment)
        assert source_fragment is not None
        self.assertEqual(source_fragment.layer, "holding")

        self.engine.remember(
            content="answer",
            actor="agent",
            reply_to=source["anchor_id"],
            source_fragment_ids=[source["id"]],
        )

        promoted = self.engine.store.get_fragment(source["id"])
        self.assertIsNotNone(promoted)
        assert promoted is not None
        self.assertEqual(promoted.layer, "working")

    def test_remember_deduplicates_agent_capture_for_same_reply_to_and_content(self) -> None:
        parent = self.engine.remember(content="question", actor="user")

        first = self.engine.remember(content="same answer", actor="agent", reply_to=parent["anchor_id"])
        second = self.engine.remember(content="same answer", actor="agent", reply_to=parent["anchor_id"])

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(first["anchor_id"], second["anchor_id"])
        self.assertEqual(len(self.engine.store.list_anchors()), 2)
        self.assertEqual(len(self.engine.store.list_fragments()), 2)

    def test_remember_deduplication_distinguishes_kind(self) -> None:
        parent = self.engine.remember(content="question", actor="user")

        regular = self.engine.remember(content="same answer", actor="agent", reply_to=parent["anchor_id"])
        policy = self.engine.remember(
            content="same answer",
            actor="agent",
            reply_to=parent["anchor_id"],
            kind="collaboration_policy",
        )

        self.assertNotEqual(regular["id"], policy["id"])
        self.assertEqual(len(self.engine.store.list_fragments()), 3)

    def test_remember_keeps_distinct_user_capture_for_same_reply_to_and_content(self) -> None:
        parent = self.engine.remember(content="question", actor="agent")

        first = self.engine.remember(content="same follow-up", actor="user", reply_to=parent["anchor_id"])
        second = self.engine.remember(content="same follow-up", actor="user", reply_to=parent["anchor_id"])

        self.assertNotEqual(first["id"], second["id"])
        self.assertNotEqual(first["anchor_id"], second["anchor_id"])
        self.assertEqual(len(self.engine.store.list_anchors()), 3)
        self.assertEqual(len(self.engine.store.list_fragments()), 3)

    def test_remember_keeps_distinct_root_user_messages_when_content_matches(self) -> None:
        first = self.engine.remember(content="same root", actor="user")
        second = self.engine.remember(content="same root", actor="user")

        self.assertNotEqual(first["id"], second["id"])
        self.assertNotEqual(first["anchor_id"], second["anchor_id"])
        self.assertEqual(len(self.engine.store.list_anchors()), 2)
        self.assertEqual(len(self.engine.store.list_fragments()), 2)

    def test_remember_keeps_distinct_agent_forks_when_content_changes(self) -> None:
        parent = self.engine.remember(content="question", actor="user")

        first = self.engine.remember(content="first answer", actor="agent", reply_to=parent["anchor_id"])
        second = self.engine.remember(content="edited answer", actor="agent", reply_to=parent["anchor_id"])

        self.assertNotEqual(first["id"], second["id"])
        self.assertNotEqual(first["anchor_id"], second["anchor_id"])
        self.assertEqual(len(self.engine.store.list_anchors()), 3)
        self.assertEqual(len(self.engine.store.list_fragments()), 3)

    def test_remember_deduplicated_agent_capture_merges_new_source_references(self) -> None:
        parent = self.engine.remember(content="question", actor="user")
        first_source = self.engine.remember(content="source one", actor="user")
        second_source = self.engine.remember(content="source two", actor="user")

        remembered = self.engine.remember(
            content="same answer",
            actor="agent",
            reply_to=parent["anchor_id"],
            source_fragment_ids=[first_source["id"]],
        )
        self.engine.remember(
            content="same answer",
            actor="agent",
            reply_to=parent["anchor_id"],
            source_fragment_ids=[second_source["id"]],
        )

        references = self.engine.store.list_references_from_anchor(remembered["anchor_id"])
        self.assertEqual(
            [reference.fragment_id for reference in references],
            [remembered["id"], first_source["id"], second_source["id"]],
        )

    def test_remember_records_unique_source_fragment_ids_only_once(self) -> None:
        source = self.engine.remember(content="source one", actor="user")

        remembered = self.engine.remember(
            content="answer with repeated sources",
            actor="agent",
            source_fragment_ids=[source["id"], source["id"], int(source["id"])],
        )

        references = self.engine.store.list_references_from_anchor(remembered["anchor_id"])
        self.assertEqual(
            [reference.fragment_id for reference in references],
            [remembered["id"], source["id"]],
        )

    def test_recent_returns_latest_root_fragments_with_filters(self) -> None:
        root = self.engine.remember(content="root", actor="user")
        older = self.engine.remember(content="older", actor="user", reply_to=root["anchor_id"])
        newest = self.engine.remember(content="newer", actor="user", reply_to=root["anchor_id"])

        result = self.engine.recent(limit=2, actor="user", reply_to=root["anchor_id"])

        self.assertEqual(result["count"], 2)
        self.assertEqual([item["id"] for item in result["items"]], [newest["id"], older["id"]])

    def test_compression_creates_child_and_moves_parent(self) -> None:
        self.engine.close()
        self.engine = make_engine(self.root / "compress", total_capacity=90)

        parent = self.engine.remember(content="x" * 40, actor="user")
        self.engine.remember(content="y" * 30, actor="user")

        fragments = self.engine.store.list_fragments_by_anchor(parent["anchor_id"])
        self.assertEqual(len(fragments), 2)
        parent_fragment = self.engine.store.get_fragment(parent["id"])
        child_fragment = next(fragment for fragment in fragments if fragment.id != parent["id"])

        assert parent_fragment is not None
        self.assertEqual(parent_fragment.layer, "holding")
        self.assertEqual(child_fragment.layer, "working")
        self.assertEqual(child_fragment.parent_id, parent_fragment.id)

        child_feedback = self.engine.store.list_feedback_for_fragment(child_fragment.id)
        child_references = self.engine.store.list_references_for_fragment(child_fragment.id)
        self.assertEqual([item.verdict for item in child_feedback], ["positive"])
        self.assertEqual(len(child_references), 2)

    def test_compress_once_retries_with_next_candidate_after_failed_attempt(self) -> None:
        self.engine.close()
        self.engine = make_engine(self.root / "compress-retry", total_capacity=400)

        first = self.engine.remember(content="x" * 60, actor="user")
        second = self.engine.remember(content="y" * 40, actor="user")
        selected_candidate = self.engine._select_compression_candidate(set())
        self.assertIsNotNone(selected_candidate)
        assert selected_candidate is not None
        fallback_candidate = next(
            fragment
            for fragment in self.engine.store.list_fragments()
            if fragment.layer == "working" and fragment.id != selected_candidate.id
        )
        self.engine.compression_backend = SelectiveCompressionBackend(
            {
                selected_candidate.content: CompressionResult(
                    content=selected_candidate.content,
                    target_length=12,
                ),
                fallback_candidate.content: CompressionResult(content="short summary", target_length=12),
            }
        )

        compressed = self.engine._compress_once({"compress": 0, "delete": 0}, set())

        self.assertTrue(compressed)
        first_fragment = self.engine.store.get_fragment(first["id"])
        second_fragment = self.engine.store.get_fragment(second["id"])
        self.assertIsNotNone(first_fragment)
        self.assertIsNotNone(second_fragment)
        assert first_fragment is not None
        assert second_fragment is not None
        failed_fragment = self.engine.store.get_fragment(selected_candidate.id)
        self.assertIsNotNone(failed_fragment)
        assert failed_fragment is not None
        self.assertEqual(failed_fragment.compression_fail_count, 1)
        self.assertEqual(failed_fragment.layer, "working")

        compressed_fragment = self.engine.store.get_fragment(fallback_candidate.id)
        self.assertIsNotNone(compressed_fragment)
        assert compressed_fragment is not None
        self.assertEqual(compressed_fragment.layer, "holding")

        child_fragment = next(
            fragment
            for fragment in self.engine.store.list_fragments_by_anchor(fallback_candidate.anchor_id)
            if fragment.id != compressed_fragment.id
        )
        self.assertEqual(child_fragment.parent_id, compressed_fragment.id)

    def test_compress_once_keeps_parent_when_result_is_not_shorter(self) -> None:
        self.engine.close()
        self.engine = make_engine(self.root / "compress-no-shorter", total_capacity=400)
        parent = self.engine.remember(content="x" * 60, actor="user")
        self.engine.compression_backend = SelectiveCompressionBackend(
            {
                "x" * 60: CompressionResult(content="x" * 60, target_length=12),
            }
        )

        compressed = self.engine._compress_once({"compress": 0, "delete": 0}, set())

        self.assertFalse(compressed)
        fragments = self.engine.store.list_fragments_by_anchor(parent["anchor_id"])
        self.assertEqual(len(fragments), 1)
        parent_fragment = self.engine.store.get_fragment(parent["id"])
        self.assertIsNotNone(parent_fragment)
        assert parent_fragment is not None
        self.assertEqual(parent_fragment.layer, "working")
        self.assertEqual(parent_fragment.compression_fail_count, 1)

    def test_compress_once_keeps_parent_when_move_cannot_complete(self) -> None:
        self.engine.close()
        self.engine = make_engine(self.root / "compress-move-fails", total_capacity=80)
        anchor_id = self.engine.store.create_anchor(None, True)
        parent_id = self.engine.store.create_fragment(
            anchor_id=anchor_id,
            parent_id=None,
            actor="user",
            kind=None,
            content="x" * 50,
            embedding_vector=None,
            layer="working",
        )
        self.engine.compression_backend = SelectiveCompressionBackend(
            {
                "x" * 50: CompressionResult(content="short summary", target_length=10),
            }
        )

        compressed = self.engine._compress_once({"compress": 0, "delete": 0}, set())

        self.assertFalse(compressed)
        fragments = self.engine.store.list_fragments_by_anchor(anchor_id)
        self.assertEqual(len(fragments), 1)
        parent_fragment = self.engine.store.get_fragment(parent_id)
        self.assertIsNotNone(parent_fragment)
        assert parent_fragment is not None
        self.assertEqual(parent_fragment.layer, "working")
        self.assertEqual(parent_fragment.compression_fail_count, 1)

    def test_maintenance_flow_converges_under_pressure(self) -> None:
        self.engine.close()
        self.engine = make_engine(self.root / "maintenance-flow", total_capacity=320)

        def make_text(label: str) -> str:
            return " ".join([label, "memory"] + ["detail" for _ in range(12)] + [f"{label}-summary"])

        for label in ["alpha", "beta", "gamma", "delta"]:
            self.engine.remember(content=make_text(label), actor="user")

        metrics = self.engine.store.list_sequence_metrics()
        stats = self.engine.stats()

        self.assertGreater(sum(metric.compress_count for metric in metrics), 0)
        self.assertGreater(sum(metric.delete_count for metric in metrics), 0)
        self.assertLessEqual(stats["working_usage"], stats["working_budget"])
        self.assertLessEqual(stats["holding_usage"], stats["holding_budget"])

    def test_compression_preserves_kind_on_child_fragment(self) -> None:
        self.engine.close()
        self.engine = make_engine(self.root / "compress-kind", total_capacity=90)

        parent = self.engine.remember(
            content="x" * 40,
            actor="agent",
            kind="collaboration_policy",
        )
        self.engine.remember(content="y" * 30, actor="user")

        fragments = self.engine.store.list_fragments_by_anchor(parent["anchor_id"])
        child_fragment = next(fragment for fragment in fragments if fragment.id != parent["id"])
        self.assertEqual(child_fragment.kind, "collaboration_policy")

    def test_compression_copies_parent_feedback_to_child_fragment(self) -> None:
        self.engine.close()
        self.engine = make_engine(self.root / "compress-feedback", total_capacity=120)

        parent = self.engine.remember(content="x" * 40, actor="user")
        feedback_source = self.engine.remember(content="feedback source", actor="user")
        self.engine.feedback(
            from_anchor_id=feedback_source["anchor_id"],
            fragment_id=parent["id"],
            verdict="negative",
        )
        self.engine.remember(content="y" * 30, actor="user")

        fragments = self.engine.store.list_fragments_by_anchor(parent["anchor_id"])
        child_fragment = next(fragment for fragment in fragments if fragment.id != parent["id"])

        child_feedback = self.engine.store.list_feedback_for_fragment(child_fragment.id)
        self.assertEqual([item.verdict for item in child_feedback], ["positive", "negative"])

    def test_purging_child_returns_failure_increment_to_parent(self) -> None:
        self.engine.close()
        self.engine = make_engine(self.root / "child-purge", total_capacity=90)

        parent = self.engine.remember(content="x" * 40, actor="user")
        self.engine.remember(content="y" * 30, actor="user")
        child = next(
            fragment
            for fragment in self.engine.store.list_fragments_by_anchor(parent["anchor_id"])
            if fragment.id != parent["id"]
        )

        self.engine.store.delete_fragment(child.id)
        parent_fragment = self.engine.store.get_fragment(parent["id"])
        self.assertIsNotNone(parent_fragment)
        assert parent_fragment is not None
        self.assertEqual(parent_fragment.compression_fail_count, 1)

    def test_delete_candidate_chooses_lowest_search_priority_in_holding(self) -> None:
        first = self.engine.remember(content="first", actor="user")
        second = self.engine.remember(content="second", actor="user")
        self.engine.feedback(
            from_anchor_id=second["anchor_id"],
            fragment_id=second["id"],
            verdict="negative",
        )
        self.engine.store.update_fragment_layer(first["id"], "holding")
        self.engine.store.update_fragment_layer(second["id"], "holding")

        candidate = self.engine._select_delete_candidate(set())
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.id, second["id"])

    def test_delete_once_removes_fragment_from_hnsw_index(self) -> None:
        backend = StubEmbeddingBackend({"first": [1.0, 0.0], "second": [0.0, 1.0]})
        ann_index = StubAnnIndex()
        self.engine.close()
        self.engine = make_engine(
            self.root / "delete-index",
            retrieval_mode="default",
            embedding_backend=backend,
            ann_index=ann_index,
        )

        first = self.engine.remember(content="first", actor="user")
        second = self.engine.remember(content="second", actor="user")
        self.engine.store.update_fragment_layer(first["id"], "holding")
        self.engine.store.update_fragment_layer(second["id"], "holding")

        deleted = self.engine._select_delete_candidate(set())
        self.assertIsNotNone(deleted)
        assert deleted is not None
        self.engine._delete_once({"compress": 0, "delete": 0}, set())

        self.assertNotIn(deleted.id, ann_index.vectors_by_fragment_id)

    def test_dynamic_layer_ratio_uses_recent_sequence_metrics(self) -> None:
        up_engine = make_engine(self.root / "ratio-up")
        down_engine = make_engine(self.root / "ratio-down")
        try:
            up_anchor = up_engine.store.create_anchor(None, True)
            up_engine.store.record_sequence_metrics(up_anchor, 10, 0, compress_count=8, delete_count=0)
            increased = up_engine._effective_working_ratio()

            down_anchor = down_engine.store.create_anchor(None, True)
            down_engine.store.record_sequence_metrics(down_anchor, 10, 0, compress_count=0, delete_count=8)
            decreased = down_engine._effective_working_ratio()
        finally:
            up_engine.close()
            down_engine.close()

        self.assertGreater(increased, self.engine.tuning.initial_working_ratio)
        self.assertLess(decreased, self.engine.tuning.initial_working_ratio)

    def test_dynamic_layer_ratio_matches_spec_formula(self) -> None:
        tuning = EngineTuning(
            initial_working_ratio=0.5,
            working_ratio_floor=0.3,
            rate_window_size=4,
        )
        self.engine.close()
        self.engine = make_engine(self.root / "ratio-spec", tuning=tuning)
        metric_pairs = [(4, 0), (2, 0), (0, 4), (0, 2)]

        expected = tuning.initial_working_ratio
        lower = tuning.working_ratio_floor
        upper = 1.0 - lower
        span = upper - lower
        running_pairs: list[tuple[int, int]] = []
        for compress_count, delete_count in metric_pairs:
            anchor_id = self.engine.store.create_anchor(None, True)
            self.engine.store.record_sequence_metrics(
                anchor_id,
                0,
                0,
                compress_count=compress_count,
                delete_count=delete_count,
            )
            running_pairs.append((compress_count, delete_count))
            window = running_pairs[-tuning.rate_window_size :]
            compress_rate = sum(pair[0] for pair in window) / tuning.rate_window_size
            delete_rate = sum(pair[1] for pair in window) / tuning.rate_window_size
            delta = compress_rate - delete_rate
            up_room = (upper - expected) / span
            down_room = (expected - lower) / span
            if delta >= 0:
                expected = expected + (up_room * delta / tuning.rate_window_size)
            else:
                expected = expected + (down_room * delta / tuning.rate_window_size)
            expected = min(upper, max(lower, expected))

        self.assertAlmostEqual(self.engine._effective_working_ratio(), expected)

    def test_fetch_supports_fragment_and_anchor_lookup(self) -> None:
        self.engine.close()
        self.engine = make_engine(self.root / "fetch", total_capacity=90)

        parent = self.engine.remember(content="x" * 40, actor="user")
        self.engine.remember(content="y" * 30, actor="user")
        by_fragment = self.engine.fetch(fragment_id=parent["id"])
        by_anchor = self.engine.fetch(anchor_id=parent["anchor_id"])

        self.assertEqual(by_fragment["count"], 1)
        self.assertEqual(by_fragment["items"][0]["id"], parent["id"])
        self.assertEqual(by_anchor["count"], 2)

    def test_fetch_by_anchor_orders_by_search_priority_then_fragment_id(self) -> None:
        anchor_id = self.engine.store.create_anchor(None, True)
        highest_id = self.engine.store.create_fragment(
            anchor_id=anchor_id,
            parent_id=None,
            actor="user",
            kind=None,
            content="highest",
            embedding_vector=None,
            layer="working",
        )
        tied_first_id = self.engine.store.create_fragment(
            anchor_id=anchor_id,
            parent_id=None,
            actor="user",
            kind=None,
            content="tied first",
            embedding_vector=None,
            layer="working",
        )
        tied_second_id = self.engine.store.create_fragment(
            anchor_id=anchor_id,
            parent_id=None,
            actor="user",
            kind=None,
            content="tied second",
            embedding_vector=None,
            layer="working",
        )

        for fragment_id, verdict in [
            (highest_id, "positive"),
            (tied_first_id, "neutral"),
            (tied_second_id, "neutral"),
        ]:
            self.engine.store.create_reference(from_anchor_id=anchor_id, fragment_id=fragment_id)
            self.engine.store.create_feedback(
                from_anchor_id=anchor_id,
                fragment_id=fragment_id,
                verdict=verdict,
            )

        result = self.engine.fetch(anchor_id=anchor_id)

        self.assertEqual(
            [item["id"] for item in result["items"]],
            [highest_id, tied_first_id, tied_second_id],
        )

    def test_read_active_collaboration_policy_uses_token_budget_with_first_item_exception(self) -> None:
        high_priority = self.engine.remember(
            content="alpha beta gamma delta epsilon zeta eta theta",
            actor="agent",
            kind="collaboration_policy",
        )
        low_priority = self.engine.remember(
            content="small policy",
            actor="agent",
            kind="collaboration_policy",
        )
        self.engine.feedback(
            from_anchor_id=low_priority["anchor_id"],
            fragment_id=low_priority["id"],
            verdict="negative",
        )

        result = self.engine.read_active_collaboration_policy(token_budget=1)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["id"], high_priority["id"])
        self.assertTrue(result["truncated"])
        self.assertGreater(result["used_token_budget"], 1)
        self.assertNotEqual(result["items"][0]["id"], low_priority["id"])

    def test_read_only_apis_do_not_record_references_or_feedback(self) -> None:
        remembered = self.engine.remember(content="alpha policy context", actor="user")
        policy = self.engine.remember(
            content="Always ask a short clarifying question when needed.",
            actor="agent",
            kind="collaboration_policy",
        )
        reference_snapshot = [
            (item.from_anchor_id, item.fragment_id)
            for item in self.engine.store.list_references()
        ]
        feedback_snapshot = [
            (item.from_anchor_id, item.fragment_id, item.verdict)
            for item in self.engine.store.list_feedback()
        ]

        search_result = self.engine.search("alpha", result_count=8, search_effort=32)
        fetch_fragment_result = self.engine.fetch(fragment_id=remembered["id"])
        fetch_anchor_result = self.engine.fetch(anchor_id=remembered["anchor_id"])
        recent_result = self.engine.recent(limit=2)
        policy_result = self.engine.read_active_collaboration_policy(token_budget=512)

        self.assertEqual(search_result["items"][0]["id"], remembered["id"])
        self.assertEqual(fetch_fragment_result["items"][0]["id"], remembered["id"])
        self.assertEqual(fetch_anchor_result["items"][0]["id"], remembered["id"])
        self.assertEqual(recent_result["count"], 2)
        self.assertEqual(policy_result["items"][0]["id"], policy["id"])
        self.assertEqual(
            [
                (item.from_anchor_id, item.fragment_id)
                for item in self.engine.store.list_references()
            ],
            reference_snapshot,
        )
        self.assertEqual(
            [
                (item.from_anchor_id, item.fragment_id, item.verdict)
                for item in self.engine.store.list_feedback()
            ],
            feedback_snapshot,
        )

    def test_stats_report_spec_fields_only(self) -> None:
        self.engine.remember(content="stats me", actor="user")

        stats = self.engine.stats()

        self.assertEqual(
            set(stats),
            {
                "fragment_count",
                "working_count",
                "holding_count",
                "working_usage",
                "holding_usage",
                "working_budget",
                "holding_budget",
                "working_ratio",
                "recent_compress_count",
                "recent_delete_count",
                "parameters",
            },
        )


if __name__ == "__main__":
    unittest.main()
