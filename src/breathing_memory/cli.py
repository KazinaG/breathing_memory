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

from .ann import HnswIndex
from .agents_template import (
    AGENTS_BLOCK_END,
    AGENTS_BLOCK_START,
    render_agents_block,
    resolve_agents_guidance_mode,
    semantic_extra_available,
)
from .config import MemoryConfig, TOTAL_CAPACITY_MB_ENV_VAR, resolve_total_capacity_mb
from .engine import BreathingMemoryEngine
from .mcp_server import serve_stdio_server
from .runtime import (
    DB_PATH_ENV_VAR,
    PROJECT_ID_ENV_VAR,
    build_project_key,
    get_app_data_root,
    resolve_db_path,
    resolve_project_identity,
)
from .store import SQLiteStore


MCP_SERVER_NAME = "breathing-memory"
MCP_SERVER_COMMAND = "breathing-memory"
MCP_SERVER_ARGS = ["serve"]
AGENTS_FILENAME = "AGENTS.md"

class CLIError(RuntimeError):
    pass


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "serve"
    try:
        if command == "serve":
            serve()
            return 0
        if command == "install-codex":
            message = install_codex_registration(total_capacity_mb=args.total_capacity_mb)
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
    install_parser = subparsers.add_parser("install-codex", help="Register the MCP server with Codex.")
    install_parser.add_argument(
        "--total-capacity-mb",
        type=_positive_float,
        help="Advanced override for the total remembered-fragment capacity used by the registered MCP server.",
    )
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
    total_capacity_mb: float | None = None,
) -> str:
    command_env = dict(os.environ if env is None else env)
    if total_capacity_mb is not None:
        command_env[TOTAL_CAPACITY_MB_ENV_VAR] = _format_total_capacity_mb(total_capacity_mb)
    working_directory = Path.cwd() if cwd is None else Path(cwd)
    registration_env = resolve_codex_registration_env(cwd=working_directory, env=command_env)
    effective_env = dict(command_env)
    effective_env.update(registration_env)
    agents_path = working_directory / AGENTS_FILENAME
    validate_agents_update_target(agents_path)
    current_agents = agents_path.read_text(encoding="utf-8") if agents_path.exists() else None
    guidance_mode = resolve_agents_guidance_mode()
    planned_agents = upsert_agents_block(current_agents, guidance_mode=guidance_mode)

    if shutil.which("codex", path=command_env.get("PATH")) is None:
        raise CLIError(
            "Codex CLI was not found on PATH. Install Codex and rerun `breathing-memory install-codex`."
        )

    existing = get_codex_registration(MCP_SERVER_NAME, runner=runner, env=command_env)
    registration_message: str
    post_check_message = ""
    if existing is not None:
        if codex_registration_matches(existing, registration_env):
            registration_message = "Codex MCP server 'breathing-memory' is already configured."
        else:
            raise CLIError(
                "Codex MCP server 'breathing-memory' already exists with a different configuration.\n"
                f"Expected: {describe_expected_registration(registration_env)}\n"
                f"Found: {describe_registration(existing)}\n"
                "Replace it with `codex mcp remove breathing-memory` and rerun `breathing-memory install-codex`."
            )
    else:
        add_command = ["codex", "mcp", "add", MCP_SERVER_NAME]
        for key, value in registration_env.items():
            add_command.extend(["--env", f"{key}={value}"])
        add_command.extend(["--", MCP_SERVER_COMMAND, *MCP_SERVER_ARGS])
        completed = runner(
            add_command,
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
            expected_env=registration_env,
            working_directory=working_directory,
        )
        if post_check.get("status") != "configured":
            raise CLIError(
                "Codex registration command completed, but the follow-up check did not confirm the expected "
                "Breathing Memory MCP entry. Rerun `breathing-memory doctor` and inspect `codex mcp get breathing-memory --json`."
            )
        post_check_message = "Post-check: Codex registration is configured."

    agents_message = write_agents_file(agents_path, current_agents, planned_agents)
    registration_binding = resolve_codex_registration_binding(cwd=working_directory, env=effective_env)
    identity_source, identity_value = resolve_project_identity(cwd=working_directory, env=effective_env)
    memory_config = resolve_memory_config(cwd=working_directory, env=effective_env)
    db_path = memory_config.db_path
    retrieval = inspect_semantic_status(memory_config)
    identity_description = describe_binding_identity(registration_binding, identity_source, identity_value)
    post_check_block = f"{post_check_message}\n" if post_check_message else ""
    next_step_lines = [
        "Next steps:",
        f"- Project identity: {identity_description}",
        f"- DB path: {db_path}",
        f"- Total capacity: {int(memory_config.total_capacity_mb * (1 << 20))} bytes ({memory_config.total_capacity_mb:g} MB)",
        f"- Effective retrieval mode: {retrieval['effective_mode']} ({retrieval['resolution_reason']})",
        "- Run `breathing-memory doctor` to verify the installation.",
        "- Open this repository in Codex and start using the registered MCP server.",
    ]
    if not retrieval["semantic_extra_available"]:
        next_step_lines.append("- Optional: install `breathing-memory[semantic]` to enable semantic retrieval.")
    elif not retrieval["hnsw_index_ready"]:
        next_step_lines.append("- Start or continue a conversation to build the HNSW index and enable `default` retrieval.")
    next_steps = "\n".join(next_step_lines)
    return f"{registration_message}\n{post_check_block}{agents_message}\n\n{next_steps}"


def inspect_memory(
    json_output: bool = False,
    runner: Any = subprocess.run,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> str:
    memory_config = resolve_inspect_memory_config(runner=runner, env=env, cwd=cwd)
    engine = BreathingMemoryEngine(config=memory_config)
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
    app_data_root = get_app_data_root()
    codex_path = shutil.which("codex", path=command_env.get("PATH"))
    environment = detect_runtime_environment(command_env)
    mount_checker = path_is_mount or (lambda path: path.is_mount())
    expected_binding = resolve_codex_registration_binding(cwd=working_directory, env=command_env)
    registration_status = inspect_codex_registration_status(
        codex_path=codex_path,
        runner=runner,
        env=command_env,
        expected_env=expected_binding["env"],
        working_directory=working_directory,
    )
    resolution_context, diagnostic_env = resolve_doctor_environment(
        base_env=command_env,
        registration_status=registration_status,
    )
    memory_config = resolve_memory_config(cwd=working_directory, env=diagnostic_env)
    identity_source, identity_value = resolve_project_identity(cwd=working_directory, env=diagnostic_env)
    db_path = memory_config.db_path
    semantic_status = inspect_semantic_status(memory_config)
    fallback_db_path = resolve_db_path(cwd=working_directory, env=command_env)
    total_capacity_mb = memory_config.total_capacity_mb
    total_capacity = int(total_capacity_mb * (1 << 20))
    app_data_root_is_mount = mount_checker(app_data_root)
    report = {
        "python_executable": sys.executable,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "working_directory": str(working_directory),
        "breathing_memory_command": shutil.which("breathing-memory", path=command_env.get("PATH")),
        "codex_command": codex_path,
        "environment": environment,
        "diagnostic_context": resolution_context,
        "project_identity": {"source": identity_source, "value": identity_value},
        "app_data_root": str(app_data_root),
        "app_data_root_is_mount": app_data_root_is_mount,
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "retrieval": semantic_status,
        "total_capacity_mb": total_capacity_mb,
        "total_capacity": total_capacity,
        "codex_registration": registration_status,
        "warnings": build_doctor_warnings(
            env=command_env,
            environment=environment,
            app_data_root_is_mount=app_data_root_is_mount,
            retrieval=semantic_status,
            registration_status=registration_status,
        ),
        "next_steps": build_doctor_next_steps(
            breathing_memory_command=shutil.which("breathing-memory", path=command_env.get("PATH")),
            codex_command=codex_path,
            registration_status=registration_status,
            db_exists=db_path.exists(),
            retrieval=semantic_status,
            fallback_db_path=fallback_db_path,
            active_db_path=db_path,
        ),
    }
    if json_output:
        return json.dumps(report, ensure_ascii=False, indent=2)
    return format_doctor_report(report)


def inspect_codex_registration_status(
    codex_path: str | None,
    runner: Any = subprocess.run,
    env: Mapping[str, str] | None = None,
    expected_env: Mapping[str, str] | None = None,
    working_directory: Path | None = None,
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
    transport = registration.get("transport")
    actual_env = extract_registration_env(registration)
    matches = codex_registration_matches(registration, expected_env or {})
    status = "configured" if matches else "conflict"
    reason = "unexpected_registration"
    if status == "configured":
        reason = "matches_expected"
    elif transport.get("env") in (None, {}) and transport.get("env_vars") in (None, []):
        reason = "legacy_unpinned_registration"
    return {
        "status": status,
        "reason": reason,
        "matches_expected": matches,
        "transport": registration.get("transport"),
        "env": actual_env,
        "working_directory": str(working_directory) if working_directory is not None else None,
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
    retrieval: Mapping[str, Any],
    registration_status: Mapping[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if environment.get("is_container") and DB_PATH_ENV_VAR not in env and not app_data_root_is_mount:
        warnings.append(
            "Container environment detected, but the default Breathing Memory app-data root is not a dedicated mount. "
            "Memory may not survive container rebuilds unless you mount this path or set BREATHING_MEMORY_DB_PATH."
        )
    if registration_status.get("reason") == "legacy_unpinned_registration":
        warnings.append(
            "Codex registration is still using the legacy unpinned format. Rerun `breathing-memory install-codex` "
            "to pin it to a stable project identity."
        )
    if retrieval.get("configured_mode") == "default" and not retrieval.get("hnsw_support_available"):
        warnings.append(
            "Configured retrieval_mode is 'default', but HNSW index support is not available in this Python environment."
        )
    if retrieval.get("configured_mode") == "lite" and not retrieval.get("semantic_extra_available"):
        warnings.append(
            "Configured retrieval_mode is 'lite', but the optional semantic extra is not available in this Python environment."
        )
    return warnings


def build_doctor_next_steps(
    breathing_memory_command: str | None,
    codex_command: str | None,
    registration_status: Mapping[str, Any],
    db_exists: bool,
    retrieval: Mapping[str, Any],
    fallback_db_path: Path,
    active_db_path: Path,
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
        if registration_status.get("reason") == "legacy_unpinned_registration":
            steps = ["Rerun `breathing-memory install-codex` to migrate Codex registration to a stable project identity."]
            if fallback_db_path.exists() and active_db_path != fallback_db_path:
                steps.append(
                    f"If you want to keep existing memory, move `{fallback_db_path}` to `{active_db_path}` manually."
                )
            return steps
        return [
            "Run `codex mcp remove breathing-memory`, then rerun `breathing-memory install-codex`."
        ]
    if status == "check_failed":
        return ["Fix the Codex registration check failure, then rerun `breathing-memory doctor`."]
    if status == "configured" and not db_exists:
        steps = ["Codex registration is pinned to a stable project identity."]
        if fallback_db_path.exists() and active_db_path != fallback_db_path:
            steps.append(
                f"If you want to keep existing memory, move `{fallback_db_path}` to `{active_db_path}` manually."
            )
        steps.append("Open this repository in Codex and start a conversation to create the project DB.")
        if not retrieval.get("semantic_extra_available"):
            steps.append("Optional: install `breathing-memory[semantic]` to enable semantic retrieval.")
        elif not retrieval.get("hnsw_index_ready"):
            steps.append("Start or continue a conversation to build the HNSW index and enable `default` retrieval.")
        return steps
    if status == "configured":
        steps = ["Codex registration is pinned to a stable project identity.", "Breathing Memory looks ready for this repository."]
        if not retrieval.get("semantic_extra_available"):
            steps.append("Optional: install `breathing-memory[semantic]` to enable semantic retrieval.")
        elif not retrieval.get("hnsw_index_ready"):
            steps.append("Start or continue a conversation to build the HNSW index and enable `default` retrieval.")
        return steps
    return []


def inspect_semantic_status(memory_config: MemoryConfig) -> dict[str, Any]:
    configured_mode = memory_config.retrieval_mode
    semantic_available = semantic_extra_available()
    hnsw_status = inspect_hnsw_status(memory_config)
    hnsw_support_available = bool(hnsw_status["support_available"])
    hnsw_index_ready = bool(hnsw_status["ready"])
    if configured_mode == "auto":
        if not semantic_available:
            effective_mode = "super_lite"
            reason = "auto_without_semantic_backend"
        elif hnsw_support_available:
            effective_mode = "default"
            reason = "auto_with_hnsw_support" if not hnsw_index_ready else "auto_with_hnsw_ready"
        else:
            effective_mode = "lite"
            reason = "auto_without_hnsw_support"
    elif configured_mode == "default":
        effective_mode = "default"
        if hnsw_index_ready:
            reason = "pinned_default_mode_ready"
        elif hnsw_support_available:
            reason = "pinned_default_mode_rebuild_required"
        else:
            reason = "pinned_default_mode_without_hnsw_support"
    elif configured_mode == "lite":
        effective_mode = "lite"
        reason = "pinned_lite_mode"
    else:
        effective_mode = "super_lite"
        reason = "pinned_super_lite_mode"
    return {
        "configured_mode": configured_mode,
        "effective_mode": effective_mode,
        "resolution_reason": reason,
        "semantic_extra_available": semantic_available,
        "hnsw_support_available": hnsw_support_available,
        "hnsw_index_ready": hnsw_index_ready,
        "hnsw_status": hnsw_status["status"],
        "hnsw_reason": hnsw_status["reason"],
        "hnsw_index_path": hnsw_status["index_path"],
        "hnsw_metadata_path": hnsw_status["metadata_path"],
        "embedding_model": memory_config.embedding_model if semantic_available else None,
    }


def inspect_hnsw_status(memory_config: MemoryConfig) -> dict[str, Any]:
    fragment_ids: list[int] = []
    if memory_config.db_path.exists():
        store = SQLiteStore(memory_config.db_path)
        try:
            fragment_ids = [fragment.id for fragment in store.list_fragments()]
        finally:
            store.close()
    return HnswIndex(memory_config.db_path).inspect(
        fragment_ids=fragment_ids,
        embedding_model=memory_config.embedding_model,
    )


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
    retrieval = report["retrieval"]
    lines = [
        f"Python: {report['python_executable']} ({report['python_version']})",
        f"Working directory: {report['working_directory']}",
        f"breathing-memory on PATH: {report['breathing_memory_command'] or 'not found'}",
        f"Codex on PATH: {report['codex_command'] or 'not found'}",
        f"Container environment: {'yes' if environment['is_container'] else 'no'}",
        f"DevContainer environment: {'yes' if environment['is_devcontainer'] else 'no'}",
        f"Diagnostic context: {report['diagnostic_context']}",
        f"Project identity: {identity['source']} = {identity['value']}",
        f"App-data root: {report['app_data_root']}",
        f"App-data root is mount: {'yes' if report['app_data_root_is_mount'] else 'no'}",
        f"DB path: {report['db_path']}",
        f"DB exists: {'yes' if report['db_exists'] else 'no'}",
        f"Configured retrieval mode: {retrieval['configured_mode']}",
        f"Effective retrieval mode: {retrieval['effective_mode']} ({retrieval['resolution_reason']})",
        f"Semantic extra available: {'yes' if retrieval['semantic_extra_available'] else 'no'}",
        f"HNSW support available: {'yes' if retrieval['hnsw_support_available'] else 'no'}",
        f"HNSW index ready: {'yes' if retrieval['hnsw_index_ready'] else 'no'} ({retrieval['hnsw_reason']})",
        f"HNSW index path: {retrieval['hnsw_index_path']}",
        f"Embedding model: {retrieval['embedding_model'] or 'not available'}",
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


def codex_registration_matches_with_env(
    registration: Mapping[str, Any],
    expected_env: Mapping[str, str],
) -> bool:
    transport = registration.get("transport")
    if not isinstance(transport, Mapping):
        return False

    return (
        transport.get("type") == "stdio"
        and transport.get("command") == MCP_SERVER_COMMAND
        and transport.get("args") == MCP_SERVER_ARGS
        and transport.get("cwd") is None
        and extract_registration_env(registration) == dict(expected_env)
        and transport.get("env_vars") in (None, [])
    )


def codex_registration_matches(registration: Mapping[str, Any], expected_env: Mapping[str, str] | None = None) -> bool:
    return codex_registration_matches_with_env(registration, expected_env or {})


def describe_expected_registration(expected_env: Mapping[str, str] | None = None) -> str:
    return describe_registration(
        {
            "transport": {
                "type": "stdio",
                "command": MCP_SERVER_COMMAND,
                "args": list(MCP_SERVER_ARGS),
                "cwd": None,
                "env": dict(expected_env or {}),
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
        extras.append(f"env={transport['env']}")
    if transport.get("env_vars"):
        extras.append(f"env_vars={transport['env_vars']}")
    extras_text = f" ({', '.join(extras)})" if extras else ""
    return f"{transport.get('type', 'unknown')} {command_line}{extras_text}"


def build_auto_project_id(identity_source: str, identity_value: str) -> str:
    return f"codex-{build_project_key(identity_source, identity_value)}"


def resolve_codex_registration_binding(
    cwd: Path,
    env: Mapping[str, str],
) -> dict[str, Any]:
    explicit_db_path = env.get(DB_PATH_ENV_VAR, "").strip()
    if explicit_db_path:
        db_path = str(Path(explicit_db_path).expanduser())
        return {
            "mode": "explicit_db_path",
            "env": {DB_PATH_ENV_VAR: db_path},
            "identity_source": "db_path",
            "identity_value": db_path,
        }

    explicit_project_id = env.get(PROJECT_ID_ENV_VAR, "").strip()
    if explicit_project_id:
        return {
            "mode": "explicit_project_id",
            "env": {PROJECT_ID_ENV_VAR: explicit_project_id},
            "identity_source": "project_id",
            "identity_value": explicit_project_id,
        }

    derived_source, derived_value = resolve_project_identity(cwd=cwd, env=env)
    auto_project_id = build_auto_project_id(derived_source, derived_value)
    return {
        "mode": "auto_project_id",
        "env": {PROJECT_ID_ENV_VAR: auto_project_id},
        "identity_source": "project_id",
        "identity_value": auto_project_id,
        "derived_source": derived_source,
        "derived_value": derived_value,
    }


def resolve_codex_registration_env(
    cwd: Path,
    env: Mapping[str, str],
) -> dict[str, str]:
    binding = resolve_codex_registration_binding(cwd=cwd, env=env)
    registration_env = dict(binding["env"])
    total_capacity_mb = env.get(TOTAL_CAPACITY_MB_ENV_VAR, "").strip()
    if total_capacity_mb:
        registration_env[TOTAL_CAPACITY_MB_ENV_VAR] = total_capacity_mb
    return registration_env


def describe_binding_identity(binding: Mapping[str, Any], identity_source: str, identity_value: str) -> str:
    if binding.get("mode") == "auto_project_id":
        return (
            f"{identity_source} = {identity_value} "
            f"(derived from {binding['derived_source']} = {binding['derived_value']})"
        )
    return f"{identity_source} = {identity_value}"


def extract_registration_env(registration: Mapping[str, Any]) -> dict[str, str]:
    transport = registration.get("transport")
    if not isinstance(transport, Mapping):
        return {}
    env = transport.get("env")
    if not isinstance(env, Mapping):
        return {}
    return {str(key): str(value) for key, value in env.items()}


def resolve_doctor_environment(
    base_env: Mapping[str, str],
    registration_status: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    registration_env = registration_status.get("env") or {}
    if registration_env:
        merged = dict(base_env)
        merged.update({str(key): str(value) for key, value in registration_env.items()})
        return ("codex_registration", merged)
    return ("working_directory", dict(base_env))


def resolve_memory_config(
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> MemoryConfig:
    return MemoryConfig(
        db_path=resolve_db_path(cwd=cwd, env=env),
        total_capacity_mb=resolve_total_capacity_mb(env=env),
    )


def resolve_inspect_memory_config(
    runner: Any = subprocess.run,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> MemoryConfig:
    command_env = dict(os.environ if env is None else env)
    if command_env.get(DB_PATH_ENV_VAR) or command_env.get(PROJECT_ID_ENV_VAR):
        return resolve_memory_config(cwd=cwd, env=command_env)

    working_directory = Path.cwd() if cwd is None else Path(cwd)
    expected_binding = resolve_codex_registration_binding(cwd=working_directory, env=command_env)
    registration_status = inspect_codex_registration_status(
        codex_path=shutil.which("codex", path=command_env.get("PATH")),
        runner=runner,
        env=command_env,
        expected_env=expected_binding["env"],
        working_directory=working_directory,
    )
    _, resolved_env = resolve_doctor_environment(
        base_env=command_env,
        registration_status=registration_status,
    )
    return resolve_memory_config(cwd=working_directory, env=resolved_env)


def _format_total_capacity_mb(value: float) -> str:
    return f"{value:g}"


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


def upsert_agents_block(current: str | None, *, guidance_mode: str) -> str:
    managed_block = render_agents_block(guidance_mode=guidance_mode)
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
