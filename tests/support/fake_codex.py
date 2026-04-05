#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def write_config(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for name, registration in state.items():
        transport = registration["transport"]
        lines.append(f"[mcp_servers.{name}]")
        lines.append(f'command = "{transport["command"]}"')
        args = ", ".join(json.dumps(arg, ensure_ascii=False) for arg in transport.get("args", []))
        lines.append(f"args = [{args}]")
        env = transport.get("env", {})
        if env:
            env_items = ", ".join(
                f"{key} = {json.dumps(value, ensure_ascii=False)}"
                for key, value in sorted(env.items())
            )
            lines.append(f"env = {{ {env_items} }}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    state_path_value = os.environ.get("FAKE_CODEX_STATE_PATH")
    if not state_path_value:
        print("FAKE_CODEX_STATE_PATH is required", file=sys.stderr)
        return 2
    state_path = Path(state_path_value)
    state = load_state(state_path)
    config_path = Path(os.environ.get("HOME", ".")) / ".codex" / "config.toml"

    if argv[:2] != ["mcp", "get"] and argv[:2] != ["mcp", "add"] and argv[:2] != ["mcp", "remove"]:
        print(f"Unsupported command: {' '.join(argv)}", file=sys.stderr)
        return 2

    if argv[:2] == ["mcp", "get"]:
        if len(argv) != 4 or argv[3] != "--json":
            print(f"Unsupported get command: {' '.join(argv)}", file=sys.stderr)
            return 2
        name = argv[2]
        registration = state.get(name)
        if registration is None:
            print(f"Error: No MCP server named '{name}' found.", file=sys.stderr)
            return 1
        print(json.dumps(registration, ensure_ascii=False))
        return 0

    if argv[:2] == ["mcp", "remove"]:
        if len(argv) != 3:
            print(f"Unsupported remove command: {' '.join(argv)}", file=sys.stderr)
            return 2
        state.pop(argv[2], None)
        save_state(state_path, state)
        write_config(config_path, state)
        return 0

    name = argv[2]
    env_pairs: dict[str, str] = {}
    idx = 3
    while idx < len(argv) and argv[idx] != "--":
        if argv[idx] != "--env" or idx + 1 >= len(argv):
            print(f"Unsupported add command: {' '.join(argv)}", file=sys.stderr)
            return 2
        key, value = argv[idx + 1].split("=", 1)
        env_pairs[key] = value
        idx += 2

    if idx >= len(argv) or argv[idx] != "--" or idx + 1 >= len(argv):
        print(f"Unsupported add command: {' '.join(argv)}", file=sys.stderr)
        return 2

    command = argv[idx + 1]
    args = argv[idx + 2 :]
    state[name] = {
        "transport": {
            "type": "stdio",
            "command": command,
            "args": args,
            "env": env_pairs,
            "env_vars": [],
            "cwd": None,
        }
    }
    save_state(state_path, state)
    write_config(config_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
