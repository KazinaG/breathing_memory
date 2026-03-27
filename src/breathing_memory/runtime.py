from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping

from platformdirs import PlatformDirs


PROJECT_ID_ENV_VAR = "BREATHING_MEMORY_PROJECT_ID"
DB_PATH_ENV_VAR = "BREATHING_MEMORY_DB_PATH"


def resolve_db_path(
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    app_data_root: Path | None = None,
) -> Path:
    environment = env if env is not None else os.environ
    override = environment.get(DB_PATH_ENV_VAR)
    if override:
        return Path(override).expanduser()

    identity_source, identity_value = resolve_project_identity(cwd=cwd, env=environment)
    project_key = build_project_key(identity_source, identity_value)
    data_root = app_data_root if app_data_root is not None else get_app_data_root()
    return data_root / "projects" / project_key / "memory.sqlite3"


def resolve_project_identity(
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    environment = env if env is not None else os.environ
    explicit_project_id = environment.get(PROJECT_ID_ENV_VAR, "").strip()
    if explicit_project_id:
        return ("project_id", explicit_project_id)

    current_directory = Path(cwd if cwd is not None else Path.cwd()).resolve()
    git_root = discover_git_root(current_directory)
    if git_root is not None:
        return ("git_root", normalize_path(git_root))
    return ("cwd", normalize_path(current_directory))


def build_project_key(identity_source: str, identity_value: str) -> str:
    display_source = identity_value if identity_source == "project_id" else Path(identity_value).name
    slug = slugify(display_source) or "project"
    digest_source = f"{identity_source}:{identity_value}"
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}"


def discover_git_root(cwd: Path) -> Path | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None

    candidate = completed.stdout.strip()
    if not candidate:
        return None
    return Path(candidate).resolve()


def get_app_data_root() -> Path:
    dirs = PlatformDirs(appname="breathing-memory", appauthor="OpenAI", ensure_exists=False)
    return Path(dirs.user_data_dir)


def normalize_path(path: Path | str) -> str:
    normalized = os.path.normcase(os.path.realpath(os.fspath(path)))
    return normalized.replace("\\", "/")


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return slug
