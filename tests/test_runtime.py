from __future__ import annotations

from pathlib import Path
import os
import stat
import subprocess
import tempfile
import unittest

from breathing_memory.cli import resolve_codex_registration_binding
from breathing_memory.runtime import build_project_key, resolve_db_path, resolve_project_identity


class RuntimeResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.app_data_root = self.root / "app-data"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_project_identity_uses_git_root_when_available(self) -> None:
        repo = self.root / "repo"
        nested = repo / "src" / "feature"
        nested.mkdir(parents=True)
        self._init_git_repo(repo)

        source, value = resolve_project_identity(cwd=nested, env={})

        self.assertEqual(source, "git_root")
        self.assertEqual(value, str(repo.resolve()))

    def test_project_identity_falls_back_to_cwd_outside_git(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()

        source, value = resolve_project_identity(cwd=outside, env={})

        self.assertEqual(source, "cwd")
        self.assertEqual(value, str(outside.resolve()))

    def test_project_id_override_takes_precedence(self) -> None:
        repo = self.root / "repo"
        repo.mkdir()

        source, value = resolve_project_identity(
            cwd=repo,
            env={"BREATHING_MEMORY_PROJECT_ID": "shared-memory"},
        )

        self.assertEqual(source, "project_id")
        self.assertEqual(value, "shared-memory")

    def test_db_path_override_bypasses_project_resolution(self) -> None:
        override = self.root / "custom" / "memory.sqlite3"

        resolved = resolve_db_path(
            cwd=self.root,
            env={"BREATHING_MEMORY_DB_PATH": str(override)},
            app_data_root=self.app_data_root,
        )

        self.assertEqual(resolved, override)

    def test_distinct_repos_resolve_to_distinct_db_paths(self) -> None:
        first_repo = self.root / "repo-one"
        second_repo = self.root / "repo-two"
        first_repo.mkdir()
        second_repo.mkdir()
        self._init_git_repo(first_repo)
        self._init_git_repo(second_repo)

        first_path = resolve_db_path(cwd=first_repo, env={}, app_data_root=self.app_data_root)
        second_path = resolve_db_path(cwd=second_repo, env={}, app_data_root=self.app_data_root)

        self.assertNotEqual(first_path, second_path)
        self.assertEqual(first_path.parent.parent, self.app_data_root / "projects")
        self.assertEqual(second_path.parent.parent, self.app_data_root / "projects")

    def test_same_repo_resolves_to_stable_db_path(self) -> None:
        repo = self.root / "repo"
        repo.mkdir()
        self._init_git_repo(repo)

        first_path = resolve_db_path(cwd=repo, env={}, app_data_root=self.app_data_root)
        second_path = resolve_db_path(cwd=repo / ".", env={}, app_data_root=self.app_data_root)

        self.assertEqual(first_path, second_path)

    def test_project_key_uses_readable_slug_and_digest(self) -> None:
        key = build_project_key("git_root", str((self.root / "My Project").resolve()))

        self.assertRegex(key, r"^my-project-[0-9a-f]{12}$")

    def test_codex_registration_binding_keeps_db_path_stable_across_cwds(self) -> None:
        repo = self.root / "repo"
        elsewhere = self.root / "elsewhere"
        repo.mkdir()
        elsewhere.mkdir()
        self._init_git_repo(repo)

        binding = resolve_codex_registration_binding(cwd=repo, env={})
        first_path = resolve_db_path(cwd=repo, env=binding["env"], app_data_root=self.app_data_root)
        second_path = resolve_db_path(cwd=elsewhere, env=binding["env"], app_data_root=self.app_data_root)

        self.assertEqual(first_path, second_path)

    def _init_git_repo(self, root: Path) -> None:
        completed = subprocess.run(
            ["git", "init"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class CodexRuntimeBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.runtime_home = self.root / "runtime"
        self.host_home = self.root / "host"
        self.workspace_codex = self.root / "workspace-codex"
        self.runtime_home.mkdir()
        self.host_home.mkdir()
        self.workspace_codex.mkdir()
        (self.host_home / "auth.json").write_text('{"token":"x"}')
        (self.workspace_codex / "config.toml").write_text('model = "gpt-5.4"\n')
        (self.workspace_codex / "README.md").write_text('# Codex Runtime\n')
        environments = self.workspace_codex / "environments"
        environments.mkdir()
        (environments / "environment.toml").write_text('name = "dev"\n')

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_bootstrap_installs_templates_over_empty_placeholders(self) -> None:
        (self.runtime_home / "config.toml").write_text("")
        (self.runtime_home / "README.md").write_text("")
        (self.runtime_home / "environments").mkdir()

        self._run_bootstrap()

        self.assertEqual((self.runtime_home / "config.toml").read_text(), 'model = "gpt-5.4"\n')
        self.assertEqual((self.runtime_home / "README.md").read_text(), '# Codex Runtime\n')
        self.assertTrue((self.runtime_home / "environments" / "environment.toml").exists())
        self.assertTrue((self.runtime_home / "auth.json").is_symlink())

    def test_bootstrap_migrates_workspace_runtime_state_when_only_templates_exist(self) -> None:
        sessions = self.workspace_codex / "sessions"
        sessions.mkdir()
        (sessions / "session.json").write_text('{"id":"1"}')

        self._run_bootstrap()

        sentinel = self.runtime_home / ".runtime_migrated_from_workspace_v1"
        self.assertTrue(sentinel.exists())
        self.assertTrue((self.runtime_home / "sessions" / "session.json").exists())
        self.assertEqual((self.runtime_home / "config.toml").read_text(), 'model = "gpt-5.4"\n')

    def test_bootstrap_preserves_existing_runtime_state(self) -> None:
        memories = self.runtime_home / "memories"
        memories.mkdir()
        (memories / "keep.txt").write_text("keep")

        self._run_bootstrap()

        self.assertTrue((memories / "keep.txt").exists())
        self.assertFalse((self.runtime_home / ".runtime_migrated_from_workspace_v1").exists())

    def _run_bootstrap(self) -> None:
        script = Path("/workspace/tools/devcontainer/bootstrap-codex-runtime.sh")
        current_mode = script.stat().st_mode
        script.chmod(current_mode | stat.S_IXUSR)
        env = os.environ.copy()
        env.update(
            {
                "CODEX_RUNTIME_HOME": str(self.runtime_home),
                "CODEX_HOST_CODEX_DIR": str(self.host_home),
                "CODEX_WORKSPACE_CODEX_DIR": str(self.workspace_codex),
            }
        )
        completed = subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
