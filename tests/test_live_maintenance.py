from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from breathing_memory.compression import CodexExecCompressionBackend
from breathing_memory.config import MemoryConfig
from breathing_memory.engine import BreathingMemoryEngine


LIVE_TEST_ENV_VAR = "BREATHING_MEMORY_RUN_LIVE_COMPRESSION_TESTS"
CODEX_PATH = shutil.which("codex")
REPO_ROOT = Path(__file__).resolve().parents[1]


class LiveMaintenanceIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir: tempfile.TemporaryDirectory[str] | None = None
        self.engine: BreathingMemoryEngine | None = None
        if os.environ.get(LIVE_TEST_ENV_VAR) != "1":
            self.skipTest(f"set {LIVE_TEST_ENV_VAR}=1 to run live compression integration tests")
        if CODEX_PATH is None:
            self.skipTest("codex is not available on PATH")

        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.command_calls: list[list[str]] = []

    def tearDown(self) -> None:
        if self.engine is not None:
            self.engine.close()
        if self.tempdir is not None:
            self.tempdir.cleanup()

    def test_live_compression_moves_parent_to_holding(self) -> None:
        self.engine = self._make_engine(total_capacity=320)

        parent = self.engine.remember(content=self._make_text("alpha", repeats=12), actor="user")
        self.engine.remember(content=self._make_text("beta", repeats=12), actor="user")

        self._assert_codex_exec_called(min_calls=1)

        fragments = self.engine.store.list_fragments_by_anchor(parent["anchor_id"])
        self.assertGreaterEqual(len(fragments), 2)

        parent_fragment = self.engine.store.get_fragment(parent["id"])
        self.assertIsNotNone(parent_fragment)
        assert parent_fragment is not None
        child_fragment = next(fragment for fragment in fragments if fragment.parent_id == parent_fragment.id)

        self.assertEqual(parent_fragment.layer, "holding")
        self.assertEqual(child_fragment.layer, "working")
        self.assertEqual(child_fragment.parent_id, parent_fragment.id)
        self.assertLess(child_fragment.content_length, parent_fragment.content_length)

        stats = self.engine.stats()
        self.assertGreater(stats["recent_compress_count"], 0)
        self.assertEqual(stats["recent_delete_count"], 0)
        self.assertLessEqual(stats["working_usage"], stats["working_budget"])
        self.assertLessEqual(stats["holding_usage"], stats["holding_budget"])

    def test_live_delete_runs_after_holding_fills(self) -> None:
        self.engine = self._make_engine(total_capacity=320)

        self.engine.remember(content=self._make_text("alpha", repeats=12), actor="user")
        self.engine.remember(content=self._make_text("beta", repeats=12), actor="user")
        self.engine.remember(content=self._make_text("gamma", repeats=12), actor="user")

        self._assert_codex_exec_called(min_calls=2)

        delete_count = sum(metric.delete_count for metric in self.engine.store.list_sequence_metrics())
        self.assertGreater(delete_count, 0)

        stats = self.engine.stats()
        self.assertGreater(stats["recent_delete_count"], 0)
        self.assertLessEqual(stats["working_usage"], stats["working_budget"])
        self.assertLessEqual(stats["holding_usage"], stats["holding_budget"])

    def test_live_newest_insert_stays_working_when_compression_runs(self) -> None:
        self.engine = self._make_engine(total_capacity=320)

        older = self.engine.remember(content=self._make_text("alpha", repeats=12), actor="user")
        newest = self.engine.remember(content=self._make_text("beta", repeats=12), actor="user")

        self._assert_codex_exec_called(min_calls=1)

        older_fragment = self.engine.store.get_fragment(older["id"])
        newest_fragment = self.engine.store.get_fragment(newest["id"])
        self.assertIsNotNone(older_fragment)
        self.assertIsNotNone(newest_fragment)
        assert older_fragment is not None
        assert newest_fragment is not None

        older_children = [
            fragment
            for fragment in self.engine.store.list_fragments_by_anchor(older["anchor_id"])
            if fragment.parent_id == older_fragment.id
        ]
        newest_fragments = self.engine.store.list_fragments_by_anchor(newest["anchor_id"])

        self.assertEqual(older_fragment.layer, "holding")
        self.assertEqual(len(older_children), 1)
        self.assertEqual(older_children[0].layer, "working")
        self.assertEqual(len(newest_fragments), 1)
        self.assertEqual(newest_fragments[0].id, newest["id"])
        self.assertEqual(newest_fragments[0].layer, "working")
        self.assertIsNone(newest_fragments[0].parent_id)

    def test_live_maintenance_flow_converges_under_pressure(self) -> None:
        self.engine = self._make_engine(total_capacity=320)

        for label, repeats in [
            ("alpha", 12),
            ("beta", 12),
            ("gamma", 12),
        ]:
            self.engine.remember(content=self._make_text(label, repeats=repeats), actor="user")

        self._assert_codex_exec_called(min_calls=2)

        metrics = self.engine.store.list_sequence_metrics()
        total_compress_count = sum(metric.compress_count for metric in metrics)
        total_delete_count = sum(metric.delete_count for metric in metrics)
        stats = self.engine.stats()

        self.assertGreater(total_compress_count, 0)
        self.assertGreater(total_delete_count, 0)
        self.assertGreater(stats["fragment_count"], 0)
        self.assertLessEqual(stats["working_usage"], stats["working_budget"])
        self.assertLessEqual(stats["holding_usage"], stats["holding_budget"])

    def test_live_compression_keeps_original_language(self) -> None:
        backend = CodexExecCompressionBackend(
            runner=self._recording_runner,
            codex_path=CODEX_PATH,
            workdir=REPO_ROOT,
        )
        original = (
            "このユーザーとは最初に方針を確認してから実装に入る。"
            "会話の流れを保ったまま、日本語で短く要約してほしい。"
        )

        result = backend.compress(original, 0.5)

        self._assert_codex_exec_called(min_calls=1)
        self.assertLess(len(result.content), len(original))
        self.assertRegex(result.content, r"[ぁ-んァ-ン一-龥]")
        self.assertTrue("方針" in result.content or "確認" in result.content)

    def _make_engine(self, total_capacity: int) -> BreathingMemoryEngine:
        config = MemoryConfig(
            db_path=self.root / "memory.sqlite3",
            total_capacity_mb=total_capacity / (1024 * 1024),
            retrieval_mode="super_lite",
        )
        return BreathingMemoryEngine(
            config=config,
            compression_backend=CodexExecCompressionBackend(
                runner=self._recording_runner,
                codex_path=CODEX_PATH,
                workdir=REPO_ROOT,
            ),
            embedding_backend=None,
        )

    def _assert_codex_exec_called(self, min_calls: int) -> None:
        self.assertGreaterEqual(len(self.command_calls), min_calls)
        for command in self.command_calls:
            self.assertEqual(command[:2], [CODEX_PATH, "exec"])
            self.assertIn("--ephemeral", command)
            self.assertIn("--output-last-message", command)

    def _recording_runner(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.command_calls.append(list(command))
        return subprocess.run(command, **kwargs)

    def _make_text(self, label: str, *, repeats: int) -> str:
        return " ".join([label, "memory"] + ["detail" for _ in range(repeats)] + [f"{label}-summary"])


if __name__ == "__main__":
    unittest.main()
