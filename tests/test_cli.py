from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from breathing_memory.cli import (
    CLIError,
    doctor,
    install_codex_registration,
    main,
    render_agents_block,
    resolve_agents_guidance_mode,
)
from breathing_memory.config import MemoryConfig
from breathing_memory.engine import BreathingMemoryEngine


class FakeRunner:
    def __init__(self, responses: list[subprocess.CompletedProcess[str]]):
        self.responses = list(responses)
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, kwargs))
        if not self.responses:
            raise AssertionError("FakeRunner received more calls than expected")
        return self.responses.pop(0)


class CodexInstallTests(unittest.TestCase):
    def test_install_codex_registers_expected_command(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            runner = FakeRunner(
                [
                    subprocess.CompletedProcess(
                        ["codex", "mcp", "get", "breathing-memory", "--json"],
                        1,
                        "",
                        "Error: No MCP server named 'breathing-memory' found.",
                    ),
                    subprocess.CompletedProcess(
                        ["codex", "mcp", "add", "breathing-memory", "--", "breathing-memory", "serve"],
                        0,
                        "",
                        "",
                    ),
                    subprocess.CompletedProcess(
                        ["codex", "mcp", "get", "breathing-memory", "--json"],
                        0,
                        (
                            '{"transport":{"type":"stdio","command":"breathing-memory","args":["serve"],'
                            '"env":null,"env_vars":[],"cwd":null}}'
                        ),
                        "",
                    ),
                ]
            )

            with patch("breathing_memory.cli.shutil.which", return_value="/usr/bin/codex"):
                message = install_codex_registration(runner=runner, env={"PATH": "/usr/bin"}, cwd=Path(tempdir))

            self.assertIn("Registered Codex MCP server 'breathing-memory'.", message)
            self.assertIn("Post-check: Codex registration is configured.", message)
            self.assertIn("Created AGENTS.md", message)
            self.assertIn("Next steps:", message)
            self.assertIn("Project identity:", message)
            self.assertIn("DB path:", message)
            self.assertIn("breathing-memory doctor", message)
            self.assertEqual(
                runner.calls[1][0],
                ["codex", "mcp", "add", "breathing-memory", "--", "breathing-memory", "serve"],
            )
            self.assertEqual(
                runner.calls[2][0],
                ["codex", "mcp", "get", "breathing-memory", "--json"],
            )
            self.assertTrue((Path(tempdir) / "AGENTS.md").exists())

    def test_existing_matching_registration_is_success(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            runner = FakeRunner(
                [
                    subprocess.CompletedProcess(
                        ["codex", "mcp", "get", "breathing-memory", "--json"],
                        0,
                        (
                            '{"transport":{"type":"stdio","command":"breathing-memory","args":["serve"],'
                            '"env":null,"env_vars":[],"cwd":null}}'
                        ),
                        "",
                    )
                ]
            )

            with patch("breathing_memory.cli.shutil.which", return_value="/usr/bin/codex"):
                first = install_codex_registration(runner=runner, env={"PATH": "/usr/bin"}, cwd=Path(tempdir))

            self.assertIn("already configured", first)
            self.assertIn("Created AGENTS.md", first)
            self.assertIn("Next steps:", first)

            runner = FakeRunner(
                [
                    subprocess.CompletedProcess(
                        ["codex", "mcp", "get", "breathing-memory", "--json"],
                        0,
                        (
                            '{"transport":{"type":"stdio","command":"breathing-memory","args":["serve"],'
                            '"env":null,"env_vars":[],"cwd":null}}'
                        ),
                        "",
                    )
                ]
            )
            with patch("breathing_memory.cli.shutil.which", return_value="/usr/bin/codex"):
                second = install_codex_registration(runner=runner, env={"PATH": "/usr/bin"}, cwd=Path(tempdir))

            self.assertIn("already configured", second)
            self.assertIn("AGENTS.md already contains", second)
            self.assertIn("Next steps:", second)
            self.assertEqual(len(runner.calls), 1)

    def test_conflicting_registration_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            runner = FakeRunner(
                [
                    subprocess.CompletedProcess(
                        ["codex", "mcp", "get", "breathing-memory", "--json"],
                        0,
                        '{"transport":{"type":"stdio","command":"python","args":["-m","breathing_memory"]}}',
                        "",
                    )
                ]
            )

            with patch("breathing_memory.cli.shutil.which", return_value="/usr/bin/codex"):
                with self.assertRaises(CLIError) as context:
                    install_codex_registration(runner=runner, env={"PATH": "/usr/bin"}, cwd=Path(tempdir))

            message = str(context.exception)
            self.assertIn("already exists with a different configuration", message)
            self.assertIn("codex mcp remove breathing-memory", message)

    def test_missing_codex_binary_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            with patch("breathing_memory.cli.shutil.which", return_value=None):
                with self.assertRaises(CLIError) as context:
                    install_codex_registration(env={"PATH": ""}, cwd=Path(tempdir))

            self.assertIn("Codex CLI was not found on PATH", str(context.exception))

    def test_existing_agents_file_keeps_external_content_and_updates_managed_block(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            agents_path = Path(tempdir) / "AGENTS.md"
            agents_path.write_text(
                "# AGENTS\n\nKeep this note.\n\n<!-- BEGIN BREATHING MEMORY -->\nold block\n<!-- END BREATHING MEMORY -->\n",
                encoding="utf-8",
            )
            runner = FakeRunner(
                [
                    subprocess.CompletedProcess(
                        ["codex", "mcp", "get", "breathing-memory", "--json"],
                        0,
                        (
                            '{"transport":{"type":"stdio","command":"breathing-memory","args":["serve"],'
                            '"env":null,"env_vars":[],"cwd":null}}'
                        ),
                        "",
                    )
                ]
            )

            with patch("breathing_memory.cli.shutil.which", return_value="/usr/bin/codex"):
                message = install_codex_registration(runner=runner, env={"PATH": "/usr/bin"}, cwd=Path(tempdir))

            self.assertIn("Updated AGENTS.md", message)
            updated = agents_path.read_text(encoding="utf-8")
            self.assertIn("Keep this note.", updated)
            self.assertIn('memory_remember(actor="user")', updated)
            self.assertIn("Keep the query in the user's language and avoid unnecessary translation.", updated)
            self.assertIn("record that with `memory_feedback`.", updated)
            self.assertEqual(updated.count("<!-- BEGIN BREATHING MEMORY -->"), 1)

    def test_render_agents_block_uses_super_lite_guidance(self) -> None:
        block = render_agents_block(guidance_mode="super_lite")

        self.assertIn("Choose a query optimized for lexical retrieval.", block)
        self.assertIn("Use keyword- or phrase-oriented queries when they improve lexical retrieval.", block)
        self.assertIn("### Feedback Attribution", block)
        self.assertIn("skip `memory_feedback` rather than guessing.", block)
        self.assertNotIn("Choose a query optimized for semantic retrieval.", block)

    def test_resolve_agents_guidance_mode_prefers_semantic_when_available_in_auto(self) -> None:
        self.assertEqual(resolve_agents_guidance_mode(retrieval_mode="auto", semantic_available=True), "semantic")
        self.assertEqual(resolve_agents_guidance_mode(retrieval_mode="auto", semantic_available=False), "super_lite")

    def test_install_codex_fails_when_agents_file_is_not_writable(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            agents_path = Path(tempdir) / "AGENTS.md"
            agents_path.write_text("existing", encoding="utf-8")
            agents_path.chmod(0o444)
            try:
                with patch("breathing_memory.cli.shutil.which", return_value="/usr/bin/codex"):
                    with self.assertRaises(CLIError) as context:
                        install_codex_registration(env={"PATH": "/usr/bin"}, cwd=Path(tempdir))
            finally:
                agents_path.chmod(0o644)

        self.assertIn("AGENTS.md is not writable", str(context.exception))

    def test_install_codex_fails_on_malformed_managed_block(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            agents_path = Path(tempdir) / "AGENTS.md"
            agents_path.write_text("# AGENTS\n\n<!-- BEGIN BREATHING MEMORY -->\nmissing end\n", encoding="utf-8")

            with patch("breathing_memory.cli.shutil.which", return_value="/usr/bin/codex"):
                with self.assertRaises(CLIError) as context:
                    install_codex_registration(env={"PATH": "/usr/bin"}, cwd=Path(tempdir))

        self.assertIn("management block is malformed", str(context.exception))

    def test_main_prints_install_error_to_stderr(self) -> None:
        stderr = io.StringIO()
        with patch("breathing_memory.cli.install_codex_registration", side_effect=CLIError("broken")):
            with contextlib.redirect_stderr(stderr):
                status = main(["install-codex"])

        self.assertEqual(status, 1)
        self.assertIn("broken", stderr.getvalue())

    def test_inspect_memory_reports_fragment_counts_and_missing_replies(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "memory.sqlite3"
            config = MemoryConfig(db_path=db_path)
            engine = BreathingMemoryEngine(config=config)
            try:
                root = engine.remember(content="hello memory", actor="user")
                engine.remember(content="hello back", actor="agent", reply_to=root["anchor_id"])
                engine.remember(content="still waiting", actor="user", reply_to=root["anchor_id"])
                engine.store.delete_fragment(root["id"])
            finally:
                engine.close()

            stdout = io.StringIO()
            env = {
                "BREATHING_MEMORY_DB_PATH": str(db_path),
            }
            with patch.dict("os.environ", env, clear=False):
                with contextlib.redirect_stdout(stdout):
                    status = main(["inspect-memory", "--json"])

            self.assertEqual(status, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["fragment_count"], 2)
            self.assertEqual(report["active_fragment_count"], 2)
            self.assertGreaterEqual(report["deleted_fragment_count"], 0)
            self.assertEqual(report["root_count"], 0)
            self.assertEqual(report["missing_reply_count"], 2)
            self.assertEqual(report["recent_fragments"][-1]["reply_target"], "missing")

    def test_doctor_reports_missing_codex_and_db_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            db_path = Path(tempdir) / "memory.sqlite3"
            stdout = io.StringIO()
            env = {
                "PATH": "",
                "BREATHING_MEMORY_DB_PATH": str(db_path),
            }
            with patch.dict("os.environ", env, clear=False):
                with contextlib.redirect_stdout(stdout):
                    status = main(["doctor", "--json"])

        self.assertEqual(status, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["db_path"], str(db_path))
        self.assertFalse(report["db_exists"])
        self.assertEqual(report["codex_registration"]["status"], "codex_not_found")
        self.assertEqual(
            report["next_steps"],
            [
                "Install Breathing Memory so `breathing-memory` is available on PATH.",
                "Install Codex and ensure `codex` is available on PATH.",
            ],
        )

    def test_doctor_uses_memory_config_capacity_instead_of_env_override(self) -> None:
        report = json.loads(
            doctor(
                json_output=True,
                env={
                    "PATH": "",
                    "BREATHING_MEMORY_TOTAL_CAPACITY": "1234",
                },
                cwd=Path("/workspace"),
            )
        )

        self.assertEqual(report["total_capacity_mb"], MemoryConfig().total_capacity_mb)
        self.assertEqual(report["total_capacity"], int(MemoryConfig().total_capacity_mb * (1 << 20)))

    def test_doctor_reports_matching_codex_registration(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            runner = FakeRunner(
                [
                    subprocess.CompletedProcess(
                        ["codex", "mcp", "get", "breathing-memory", "--json"],
                        0,
                        (
                            '{"transport":{"type":"stdio","command":"breathing-memory","args":["serve"],'
                            '"env":null,"env_vars":[],"cwd":null}}'
                        ),
                        "",
                    )
                ]
            )

            with patch("breathing_memory.cli.shutil.which", side_effect=lambda name, path=None: "/usr/bin/codex" if name == "codex" else "/usr/bin/breathing-memory"):
                report = json.loads(
                    doctor(
                        json_output=True,
                        runner=runner,
                        env={"PATH": "/usr/bin"},
                        cwd=Path(tempdir),
                    )
                )

        self.assertEqual(report["codex_registration"]["status"], "configured")
        self.assertTrue(report["codex_registration"]["matches_expected"])
        self.assertEqual(
            report["next_steps"],
            ["Open this repository in Codex and start a conversation to create the project DB."],
        )

    def test_doctor_warns_when_container_app_data_is_not_mounted(self) -> None:
        report = json.loads(
            doctor(
                json_output=True,
                env={"PATH": "", "DEVCONTAINER": "1"},
                cwd=Path("/workspace"),
                path_is_mount=lambda path: False,
            )
        )

        self.assertTrue(report["environment"]["is_container"])
        self.assertTrue(report["environment"]["is_devcontainer"])
        self.assertTrue(report["warnings"])
        self.assertIn("Memory may not survive container rebuilds", report["warnings"][0])

    def test_python_module_entrypoint_starts_server(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            env = {
                "PYTHONPATH": "/workspace/public/src",
                "BREATHING_MEMORY_DB_PATH": str(Path(tempdir) / "memory.sqlite3"),
            }
            with patch.dict("os.environ", env, clear=False):
                output = self._round_trip_module_process()

        self.assertEqual(output["initialize_protocol"], output["latest_protocol"])
        self.assertEqual(
            output["tool_names"],
            ["memory_remember", "memory_search", "memory_fetch", "memory_feedback", "memory_stats"],
        )
        self.assertEqual(output["stats_fragment_count"], 0)

    def _round_trip_module_process(self) -> dict[str, object]:
        script = """
import anyio
import json
import os
import sys

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main():
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "breathing_memory"],
        env=dict(os.environ),
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            stats = await session.call_tool("memory_stats", {})

            print(
                json.dumps(
                    {
                        "initialize_protocol": init.protocolVersion,
                        "latest_protocol": types.LATEST_PROTOCOL_VERSION,
                        "tool_names": [tool.name for tool in tools.tools],
                        "stats_fragment_count": stats.structuredContent["fragment_count"],
                    }
                )
            )


anyio.run(main)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return dict(__import__("json").loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
