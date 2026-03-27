from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .config import MemoryConfig
from .engine import BreathingMemoryEngine
from .mcp_server import serve_stdio_server
from .runtime import DB_PATH_ENV_VAR, get_app_data_root, resolve_db_path, resolve_project_identity


MCP_SERVER_NAME = "breathing-memory"
MCP_SERVER_COMMAND = "breathing-memory"
MCP_SERVER_ARGS = ["serve"]
AGENTS_FILENAME = "AGENTS.md"
AGENTS_BLOCK_START = "<!-- BEGIN BREATHING MEMORY -->"
AGENTS_BLOCK_END = "<!-- END BREATHING MEMORY -->"
AGENTS_BLOCK_BODY = """## Breathing Memory

This repository uses Breathing Memory during Codex work.

### Required Flow

For every user turn in this repository, Codex must use the MCP tools in this order:

1. If the immediately previous final agent answer has not yet been remembered, save it first with `memory_remember(actor="agent")`
2. Save the current user message with `memory_remember(actor="user")`
3. Search with `memory_search` before producing the next final answer

Use the returned previous-agent `anchor_id` as the current user's `reply_to` when the user is replying to the immediately previous answer.
When the user is replying or forking from an earlier remembered anchor, pass that target as the user's `reply_to` instead.
For a root user message, omit `reply_to`.

### What To Save

- Save every user message.
- Save each final user-facing answer on the next user turn.
- Do not save intermediary commentary, progress updates, or tool-status messages.
- Do not save duplicate retries of the same final answer.
- `memory_remember` suppresses duplicate deferred agent capture for the same `reply_to` and content, but callers must still pass accurate `reply_to` values and capture timing.
- If no later user turn arrives, the final agent answer may remain unremembered.

### Search Query

- Use the latest user message itself as the default `memory_search.query`.
- When retrieval quality would clearly benefit, an agent-authored query is also allowed.
- Keep the query in the user's language and avoid unnecessary translation or paraphrase.
- Use the default `search_effort` of `32` unless there is a concrete reason to choose a different valid value up front.
- Start with a `result_count` of `8` unless there is a concrete reason to choose a different valid value up front.
- The MCP-calling agent may begin with broader retrieval such as `result_count = 16` or higher `search_effort` when the query clearly warrants it.
- If the first search result looks insufficient, rerun `memory_search` with a broader `result_count`, a higher `search_effort`, or both.
- Treat `result_count` as powers of two from the base `8`, and `search_effort` as powers of two from the base `32`.
- Treat semantic retrieval modes as `super_lite` (lexical only), `lite` (embedding without ANN index), and the default mode (embedding with HNSW).

### Source References

- If the deferred final answer materially uses fragments returned by `memory_search`, pass those fragment ids as `source_fragment_ids` when that answer is persisted on the next user turn.
- If no search result materially informed the final answer, omit `source_fragment_ids`.

### Failure Policy

- Do not fabricate remembered ids such as `reply_to` or `source_fragment_ids`.
- If semantic-index mode is enabled and the semantic index is being rebuilt or recovered, do not issue other Breathing Memory mutations or semantic searches until that rebuild completes.
- `archived_sessions/*.jsonl` and other Codex runtime files are not the primary capture path. They are internal implementation details and must not be used as the default memory source.
"""


class CLIError(RuntimeError):
    pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "serve"
    try:
        if command == "serve":
            serve()
            return 0
        if command == "install-codex":
            message = install_codex_registration()
            print(message)
            return 0
        if command == "inspect-memory":
            report = inspect_memory(json_output=args.json)
            print(report)
            return 0
        if command == "doctor":
            report = doctor(json_output=args.json)
            print(report)
            return 0
    except CLIError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser.error(f"unknown command: {command}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="breathing-memory",
        description="Breathing Memory MCP server and Codex integration helpers.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="Run the MCP server over stdio.")
    subparsers.add_parser("install-codex", help="Register the MCP server with Codex.")
    inspect_parser = subparsers.add_parser(
        "inspect-memory",
        help="Inspect remembered fragments with a compact diagnostic report.",
    )
    inspect_parser.add_argument("--json", action="store_true", help="Print the memory report as JSON.")
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Inspect local Breathing Memory and Codex installation state.",
    )
    doctor_parser.add_argument("--json", action="store_true", help="Print the diagnostic report as JSON.")
    return parser


def serve() -> None:
    asyncio.run(serve_stdio_server(config=MemoryConfig()))


def install_codex_registration(
    runner: Any = subprocess.run,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> str:
    command_env = dict(os.environ if env is None else env)
    working_directory = Path.cwd() if cwd is None else Path(cwd)
    agents_path = working_directory / AGENTS_FILENAME
    validate_agents_update_target(agents_path)
    current_agents = agents_path.read_text(encoding="utf-8") if agents_path.exists() else None
    planned_agents = upsert_agents_block(current_agents)

    if shutil.which("codex", path=command_env.get("PATH")) is None:
        raise CLIError(
            "Codex CLI was not found on PATH. Install Codex and rerun `breathing-memory install-codex`."
        )

    existing = get_codex_registration(MCP_SERVER_NAME, runner=runner, env=command_env)
    registration_message: str
    post_check_message = ""
    if existing is not None:
        if codex_registration_matches(existing):
            registration_message = "Codex MCP server 'breathing-memory' is already configured."
        else:
            raise CLIError(
                "Codex MCP server 'breathing-memory' already exists with a different configuration.\n"
                f"Expected: {describe_expected_registration()}\n"
                f"Found: {describe_registration(existing)}\n"
                "Replace it with `codex mcp remove breathing-memory` and rerun `breathing-memory install-codex`."
            )
    else:
        completed = runner(
            ["codex", "mcp", "add", MCP_SERVER_NAME, "--", MCP_SERVER_COMMAND, *MCP_SERVER_ARGS],
            capture_output=True,
            text=True,
            check=False,
            env=command_env,
        )
        if completed.returncode != 0:
            raise CLIError(format_subprocess_error("Failed to register the MCP server with Codex.", completed))
        registration_message = "Registered Codex MCP server 'breathing-memory'."
        post_check = inspect_codex_registration_status(
            codex_path=shutil.which("codex", path=command_env.get("PATH")),
            runner=runner,
            env=command_env,
        )
        if post_check.get("status") != "configured":
            raise CLIError(
                "Codex registration command completed, but the follow-up check did not confirm the expected "
                "Breathing Memory MCP entry. Rerun `breathing-memory doctor` and inspect `codex mcp get breathing-memory --json`."
            )
        post_check_message = "Post-check: Codex registration is configured."

    agents_message = write_agents_file(agents_path, current_agents, planned_agents)
    identity_source, identity_value = resolve_project_identity(cwd=working_directory, env=command_env)
    db_path = resolve_db_path(cwd=working_directory, env=command_env)
    post_check_block = f"{post_check_message}\n" if post_check_message else ""
    next_steps = "\n".join(
        [
            "Next steps:",
            f"- Project identity: {identity_source} = {identity_value}",
            f"- DB path: {db_path}",
            "- Run `breathing-memory doctor` to verify the installation.",
            "- Open this repository in Codex and start using the registered MCP server.",
        ]
    )
    return f"{registration_message}\n{post_check_block}{agents_message}\n\n{next_steps}"


def inspect_memory(
    json_output: bool = False,
) -> str:
    engine = BreathingMemoryEngine(config=MemoryConfig())
    try:
        report = build_memory_report(engine)
    finally:
        engine.close()
    if json_output:
        return json.dumps(report, ensure_ascii=False, indent=2)
    return format_memory_summary(report)


def doctor(
    json_output: bool = False,
    runner: Any = subprocess.run,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    path_is_mount: Any = None,
) -> str:
    command_env = dict(os.environ if env is None else env)
    working_directory = Path.cwd() if cwd is None else Path(cwd)
    identity_source, identity_value = resolve_project_identity(cwd=working_directory, env=command_env)
    db_path = resolve_db_path(cwd=working_directory, env=command_env)
    app_data_root = get_app_data_root()
    codex_path = shutil.which("codex", path=command_env.get("PATH"))
    environment = detect_runtime_environment(command_env)
    mount_checker = path_is_mount or (lambda path: path.is_mount())
    registration_status = inspect_codex_registration_status(
        codex_path=codex_path,
        runner=runner,
        env=command_env,
    )
    total_capacity_mb = MemoryConfig(db_path=db_path).total_capacity_mb
    total_capacity = int(total_capacity_mb * (1 << 20))
    app_data_root_is_mount = mount_checker(app_data_root)
    report = {
        "python_executable": sys.executable,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "working_directory": str(working_directory),
        "breathing_memory_command": shutil.which("breathing-memory", path=command_env.get("PATH")),
        "codex_command": codex_path,
        "environment": environment,
        "project_identity": {"source": identity_source, "value": identity_value},
        "app_data_root": str(app_data_root),
        "app_data_root_is_mount": app_data_root_is_mount,
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "total_capacity_mb": total_capacity_mb,
        "total_capacity": total_capacity,
        "codex_registration": registration_status,
        "warnings": build_doctor_warnings(
            env=command_env,
            environment=environment,
            app_data_root_is_mount=app_data_root_is_mount,
        ),
        "next_steps": build_doctor_next_steps(
            breathing_memory_command=shutil.which("breathing-memory", path=command_env.get("PATH")),
            codex_command=codex_path,
            registration_status=registration_status,
            db_exists=db_path.exists(),
        ),
    }
    if json_output:
        return json.dumps(report, ensure_ascii=False, indent=2)
    return format_doctor_report(report)


def inspect_codex_registration_status(
    codex_path: str | None,
    runner: Any = subprocess.run,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if codex_path is None:
        return {"status": "codex_not_found", "matches_expected": False}
    try:
        registration = get_codex_registration(MCP_SERVER_NAME, runner=runner, env=env)
    except CLIError as exc:
        return {
            "status": "check_failed",
            "matches_expected": False,
            "error": str(exc),
        }
    if registration is None:
        return {"status": "missing", "matches_expected": False}
    return {
        "status": "configured" if codex_registration_matches(registration) else "conflict",
        "matches_expected": codex_registration_matches(registration),
        "transport": registration.get("transport"),
    }


def detect_runtime_environment(env: Mapping[str, str]) -> dict[str, bool]:
    is_container = any(Path(path).exists() for path in ("/.dockerenv", "/run/.containerenv"))
    is_devcontainer = (
        bool(env.get("REMOTE_CONTAINERS"))
        or bool(env.get("DEVCONTAINER"))
        or (is_container and bool(env.get("VSCODE_IPC_HOOK_CLI")))
    )
    return {
        "is_container": is_container,
        "is_devcontainer": is_devcontainer,
    }


def build_doctor_warnings(
    env: Mapping[str, str],
    environment: Mapping[str, bool],
    app_data_root_is_mount: bool,
) -> list[str]:
    warnings: list[str] = []
    if environment.get("is_container") and DB_PATH_ENV_VAR not in env and not app_data_root_is_mount:
        warnings.append(
            "Container environment detected, but the default Breathing Memory app-data root is not a dedicated mount. "
            "Memory may not survive container rebuilds unless you mount this path or set BREATHING_MEMORY_DB_PATH."
        )
    return warnings


def build_doctor_next_steps(
    breathing_memory_command: str | None,
    codex_command: str | None,
    registration_status: Mapping[str, Any],
    db_exists: bool,
) -> list[str]:
    steps: list[str] = []
    if breathing_memory_command is None:
        steps.append("Install Breathing Memory so `breathing-memory` is available on PATH.")
    if codex_command is None:
        steps.append("Install Codex and ensure `codex` is available on PATH.")
    if steps:
        return steps

    status = registration_status.get("status")
    if status == "missing":
        return ["Run `breathing-memory install-codex` in this repository."]
    if status == "conflict":
        return [
            "Run `codex mcp remove breathing-memory`, then rerun `breathing-memory install-codex`."
        ]
    if status == "check_failed":
        return ["Fix the Codex registration check failure, then rerun `breathing-memory doctor`."]
    if status == "configured" and not db_exists:
        return ["Open this repository in Codex and start a conversation to create the project DB."]
    if status == "configured":
        return ["Breathing Memory looks ready for this repository."]
    return []


def build_memory_report(engine: BreathingMemoryEngine) -> dict[str, Any]:
    store = engine.store
    fragments = store.list_fragments()
    anchors = {anchor.id: anchor for anchor in store.list_anchors()}
    recent_fragment_items = []
    missing_reply_count = 0
    active_root_count = 0
    for fragment in fragments:
        anchor = anchors.get(fragment.anchor_id)
        reply_to = None if anchor is None else anchor.replies_to_anchor_id
        is_root = False if anchor is None else anchor.is_root
        reply_missing = reply_to is None and not is_root
        if reply_missing:
            missing_reply_count += 1
        if is_root:
            active_root_count += 1
        recent_fragment_items.append(
            {
                "id": fragment.id,
                "anchor_id": fragment.anchor_id,
                "parent_id": fragment.parent_id,
                "reply_to": reply_to,
                "reply_target": (
                    "missing"
                    if reply_missing
                    else (None if reply_to is None else (reply_to if reply_to in anchors else f"missing({reply_to})"))
                ),
                "actor": fragment.actor,
                "layer": fragment.layer,
                "content_length": fragment.content_length,
            }
        )
    recent_fragment_items = recent_fragment_items[-10:]

    deleted_fragment_count = sum(metric.delete_count for metric in store.list_sequence_metrics())
    stats = engine.stats()
    return {
        "fragment_count": len(fragments),
        "active_fragment_count": len(fragments),
        "deleted_fragment_count": deleted_fragment_count,
        "working_count": stats["working_count"],
        "holding_count": stats["holding_count"],
        "root_count": active_root_count,
        "missing_reply_count": missing_reply_count,
        "reference_log_count": len(store.list_references()),
        "feedback_log_count": len(store.list_feedback()),
        "recent_fragments": recent_fragment_items,
    }


def format_memory_summary(report: Mapping[str, Any]) -> str:
    return " ".join(
        [
            f"fragments={report['fragment_count']}",
            f"active={report['active_fragment_count']}",
            f"deleted={report['deleted_fragment_count']}",
            f"working={report['working_count']}",
            f"holding={report['holding_count']}",
            f"roots={report['root_count']}",
            f"missing_reply={report['missing_reply_count']}",
        ]
    )


def format_doctor_report(report: Mapping[str, Any]) -> str:
    registration = report["codex_registration"]
    environment = report["environment"]
    identity = report["project_identity"]
    lines = [
        f"Python: {report['python_executable']} ({report['python_version']})",
        f"Working directory: {report['working_directory']}",
        f"breathing-memory on PATH: {report['breathing_memory_command'] or 'not found'}",
        f"Codex on PATH: {report['codex_command'] or 'not found'}",
        f"Container environment: {'yes' if environment['is_container'] else 'no'}",
        f"DevContainer environment: {'yes' if environment['is_devcontainer'] else 'no'}",
        f"Project identity: {identity['source']} = {identity['value']}",
        f"App-data root: {report['app_data_root']}",
        f"App-data root is mount: {'yes' if report['app_data_root_is_mount'] else 'no'}",
        f"DB path: {report['db_path']}",
        f"DB exists: {'yes' if report['db_exists'] else 'no'}",
        f"Total capacity: {report['total_capacity']} bytes",
        f"Codex registration: {registration['status']}",
    ]
    if registration.get("error"):
        lines.append(f"Registration error: {registration['error']}")
    for warning in report["warnings"]:
        lines.append(f"Warning: {warning}")
    for step in report["next_steps"]:
        lines.append(f"Next step: {step}")
    return "\n".join(lines)


def get_codex_registration(
    name: str,
    runner: Any = subprocess.run,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    completed = runner(
        ["codex", "mcp", "get", name, "--json"],
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ if env is None else env),
    )
    if completed.returncode == 0:
        return json.loads(completed.stdout)

    combined_output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
    if f"No MCP server named '{name}' found." in combined_output:
        return None
    raise CLIError(format_subprocess_error("Failed to inspect Codex MCP registrations.", completed))


def codex_registration_matches(registration: Mapping[str, Any]) -> bool:
    transport = registration.get("transport")
    if not isinstance(transport, Mapping):
        return False

    return (
        transport.get("type") == "stdio"
        and transport.get("command") == MCP_SERVER_COMMAND
        and transport.get("args") == MCP_SERVER_ARGS
        and transport.get("cwd") is None
        and transport.get("env") in (None, {})
        and transport.get("env_vars") in (None, [])
    )


def describe_expected_registration() -> str:
    return describe_registration(
        {
            "transport": {
                "type": "stdio",
                "command": MCP_SERVER_COMMAND,
                "args": list(MCP_SERVER_ARGS),
                "cwd": None,
                "env": None,
                "env_vars": [],
            }
        }
    )


def describe_registration(registration: Mapping[str, Any]) -> str:
    transport = registration.get("transport")
    if not isinstance(transport, Mapping):
        return "unknown transport"

    command = transport.get("command")
    args = transport.get("args") or []
    command_line = shlex.join([str(command), *[str(arg) for arg in args]])
    extras = []
    if transport.get("cwd"):
        extras.append(f"cwd={transport['cwd']}")
    if transport.get("env"):
        extras.append("env=custom")
    if transport.get("env_vars"):
        extras.append(f"env_vars={transport['env_vars']}")
    extras_text = f" ({', '.join(extras)})" if extras else ""
    return f"{transport.get('type', 'unknown')} {command_line}{extras_text}"


def format_subprocess_error(prefix: str, completed: Any) -> str:
    details = []
    stdout = getattr(completed, "stdout", "")
    stderr = getattr(completed, "stderr", "")
    if stdout and stdout.strip():
        details.append(f"stdout: {stdout.strip()}")
    if stderr and stderr.strip():
        details.append(f"stderr: {stderr.strip()}")
    if details:
        return f"{prefix}\n" + "\n".join(details)
    return prefix


def validate_agents_update_target(agents_path: Path) -> None:
    parent = agents_path.parent
    if not parent.exists():
        raise CLIError(f"Current working directory does not exist: {parent}")
    if agents_path.exists():
        if not agents_path.is_file():
            raise CLIError(f"{AGENTS_FILENAME} exists but is not a regular file: {agents_path}")
        if not os.access(agents_path, os.W_OK):
            raise CLIError(f"{AGENTS_FILENAME} is not writable: {agents_path}")
    elif not os.access(parent, os.W_OK):
        raise CLIError(f"Cannot create {AGENTS_FILENAME} in the current working directory: {parent}")


def write_agents_file(agents_path: Path, current: str | None, updated: str) -> str:
    if current == updated:
        return f"{AGENTS_FILENAME} already contains the Breathing Memory instructions."
    agents_path.write_text(updated, encoding="utf-8")
    if current is None:
        return f"Created {AGENTS_FILENAME} with Breathing Memory instructions."
    return f"Updated {AGENTS_FILENAME} with Breathing Memory instructions."


def upsert_agents_block(current: str | None) -> str:
    managed_block = render_agents_block()
    if current is None or not current.strip():
        return "# AGENTS\n\n" + managed_block + "\n"

    start_index = current.find(AGENTS_BLOCK_START)
    end_index = current.find(AGENTS_BLOCK_END)
    if start_index == -1 and end_index == -1:
        base = current.rstrip()
        return f"{base}\n\n{managed_block}\n"
    if start_index == -1 or end_index == -1 or end_index < start_index:
        raise CLIError(
            f"Failed to update {AGENTS_FILENAME} safely because the Breathing Memory management block is malformed."
        )

    end_index += len(AGENTS_BLOCK_END)
    prefix = current[:start_index].rstrip()
    suffix = current[end_index:].lstrip()
    sections = [section for section in [prefix, managed_block, suffix] if section]
    return "\n\n".join(sections) + "\n"


def render_agents_block() -> str:
    return f"{AGENTS_BLOCK_START}\n{AGENTS_BLOCK_BODY.rstrip()}\n{AGENTS_BLOCK_END}"
