from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


class InstallCodexIntegrationTests(unittest.TestCase):
    def test_install_codex_repo_flow_works_via_real_subprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            repo = root / "repo"
            repo.mkdir()
            self._init_git_repo(repo)

            bin_dir = root / "bin"
            bin_dir.mkdir()
            state_path = root / "codex-state.json"
            venv_dir = root / "venv"
            self._create_virtualenv(venv_dir)
            venv_python = self._venv_python(venv_dir)
            venv_bin = venv_python.parent
            codex_script = bin_dir / "codex"
            helper = Path(__file__).with_name("support") / "fake_codex.py"
            codex_script.write_text(
                "#!/usr/bin/env bash\n"
                f'exec "{sys.executable}" "{helper}" "$@"\n',
                encoding="utf-8",
            )
            codex_script.chmod(codex_script.stat().st_mode | stat.S_IXUSR)

            env = {
                "HOME": str(root / "home"),
                "PATH": f"{bin_dir}:{venv_bin}:{self._current_path()}",
                "FAKE_CODEX_STATE_PATH": str(state_path),
            }
            Path(env["HOME"]).mkdir()

            install = subprocess.run(
                [str(venv_python), "-m", "breathing_memory", "install-codex", "--codex-config", "repo"],
                cwd=repo,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(install.returncode, 0, install.stderr)
            self.assertIn("Registered Codex MCP server 'breathing-memory'.", install.stdout)
            self.assertTrue((repo / ".codex" / "config.toml").exists())
            self.assertTrue((repo / "AGENTS.md").exists())

            doctor = subprocess.run(
                [str(venv_python), "-m", "breathing_memory", "doctor", "--json"],
                cwd=repo,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(doctor.returncode, 0, doctor.stderr)
            report = json.loads(doctor.stdout)
            self.assertEqual(report["codex_registration"]["status"], "configured")
            self.assertEqual(report["codex_registration"]["source"], "repo_local")

            registration = subprocess.run(
                ["codex", "mcp", "get", "breathing-memory", "--json"],
                cwd=repo,
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(registration.returncode, 0, registration.stderr)
            payload = json.loads(registration.stdout)
            self.assertEqual(payload["transport"]["command"], "breathing-memory")
            self.assertEqual(payload["transport"]["args"], ["serve"])

    def _init_git_repo(self, root: Path) -> None:
        completed = subprocess.run(
            ["git", "init"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def _create_virtualenv(self, root: Path) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "venv", str(root)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        completed = subprocess.run(
            [str(self._venv_python(root)), "-m", "pip", "install", "-e", "/workspace/public"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def _venv_python(self, root: Path) -> Path:
        return root / "bin" / "python"

    def _current_path(self) -> str:
        return os.environ.get("PATH", "")


if __name__ == "__main__":
    unittest.main()
