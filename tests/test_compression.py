from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from breathing_memory.compression import CodexExecCompressionBackend


class CodexExecCompressionBackendTests(unittest.TestCase):
    def test_falls_back_to_stub_when_codex_is_missing(self) -> None:
        backend = CodexExecCompressionBackend()

        with patch("breathing_memory.compression.shutil.which", return_value=None):
            result = backend.compress("alpha beta gamma delta", 0.8)

        self.assertTrue(result.content)
        self.assertLessEqual(len(result.content), len("alpha beta gamma delta"))

    def test_uses_ephemeral_codex_exec_and_reads_last_message(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            output_index = command.index("--output-last-message") + 1
            output_path = Path(command[output_index])
            output_path.write_text("compressed core", encoding="utf-8")
            self.assertIn("--ephemeral", command)
            self.assertEqual(kwargs["text"], True)
            self.assertIn("Highest priority: minimize character count aggressively.", kwargs["input"])
            return subprocess.CompletedProcess(command, 0, "", "")

        backend = CodexExecCompressionBackend(runner=fake_runner, codex_path="/usr/bin/codex")

        with tempfile.TemporaryDirectory() as tempdir:
            result = backend.compress("original content with extra detail", 0.8)

        self.assertEqual(result.content, "compressed core")
        self.assertEqual(calls[0][0:3], ["/usr/bin/codex", "exec", "--ephemeral"])

