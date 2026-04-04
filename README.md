# Breathing Memory

Breathing Memory is a local memory support system for coding agents. It exposes a stable stdio MCP server for agent clients, also exposes a typed in-process core API for non-MCP consumers, stores memory in SQLite, and isolates memory by project so one installation can be reused across repositories without mixing contexts.

## Overview

Breathing Memory keeps collaboration context that an agent should remember but a repository should not need to encode everywhere.

- local stdio MCP server
- SQLite storage under user app-data, isolated by project
- fragment-centric public model built around `anchor` and `fragment`
- text-first retrieval today, with a public search surface already aligned for later semantic retrieval work
- dynamic `working / holding` maintenance with a compression backend that uses a supported coding agent without polluting normal conversation history

## Supported Clients

- Codex
- Python in-process consumers through `breathing_memory.core`

## Installation

The intended long-term user path is:

```bash
pip install 'breathing-memory[semantic]'
breathing-memory install-codex
```

`breathing-memory install-codex` registers the `breathing-memory` MCP server with the currently supported client, pins that registration to a stable project identity for the current repository, and creates or updates the managed Breathing Memory block in the current repository's `AGENTS.md`.
The default path is the user-level Codex config. If you want repository-local Codex config instead, choose it explicitly with `breathing-memory install-codex --codex-config repo`.
After upgrading the package, rerun `breathing-memory install-codex` to refresh the managed Breathing Memory block in `AGENTS.md`.

Published package:

- recommended: `pip install 'breathing-memory[semantic]'`
- minimal `super_lite` install: `pip install breathing-memory`
- contributor setup and unreleased local work: [docs/dev-guide.md](docs/dev-guide.md)

## Quickstart

Recommended first run:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install 'breathing-memory[semantic]'
breathing-memory doctor
breathing-memory install-codex
```

Useful commands:

- `breathing-memory doctor`: inspect installation, active project identity, DB path selection, Codex registration state, and effective retrieval mode
- `breathing-memory serve`: start the stdio MCP server
- `breathing-memory warmup`: eagerly load the semantic embedding backend for the current environment
- `breathing-memory inspect-memory --json`: inspect current memory state
- after `pip install -U ...`, rerun `breathing-memory install-codex` to refresh the managed `AGENTS.md` guidance

Codex registration targets:

- `breathing-memory install-codex`: write to the user-level Codex config
- `breathing-memory install-codex --codex-config repo`: write to `.codex/config.toml` in the current repository

Environment-specific setup:

- normal local environment
  - `pip install 'breathing-memory[semantic]'`
  - `breathing-memory doctor`
  - `breathing-memory install-codex`
- repository-local Codex config, including DevContainer workflows that track `.codex/config.toml`
  - `pip install 'breathing-memory[semantic]'`
  - `breathing-memory doctor`
  - `breathing-memory install-codex --codex-config repo`
- slim containers where native build dependencies may be missing
  - install Python headers and compiler toolchain first
  - then run `pip install 'breathing-memory[semantic]'`
  - confirm the effective mode with `breathing-memory doctor`
- minimal lexical-only install
  - `pip install breathing-memory`
  - `breathing-memory install-codex`
- environments where you want to preload the semantic model before a session
  - `pip install 'breathing-memory[semantic]'`
  - `breathing-memory warmup`

## Usage Surfaces

Breathing Memory keeps the MCP surface stable while also supporting direct in-process use.

Use MCP when your agent runtime expects tools over stdio:

```bash
breathing-memory serve
```

Use the typed core API when your runtime wants to call memory directly in Python:

```python
from breathing_memory.core import (
    ReadActiveCollaborationPolicyRequest,
    RememberRequest,
    SearchRequest,
    create_engine,
)

engine = create_engine()
engine.remember(RememberRequest(content="hello", actor="user"))
engine.search(SearchRequest(query="hello"))
engine.read_active_collaboration_policy(
    ReadActiveCollaborationPolicyRequest(token_budget=512)
)
engine.close()
```

If you need to inspect or override the resolved project-scoped config first, call `breathing_memory.core.resolve_memory_config(...)` and pass the result to `create_engine(...)`.

## How Memory Works

Breathing Memory does not auto-capture the full client conversation by itself. The consuming runtime must call it explicitly, either through the MCP tools or through the typed core API.

At a high level:

1. save turns explicitly
2. search before answering
3. record references and feedback only when they materially apply

The detailed MCP caller flow, duplicate-handling rules, and attribution rules live in [docs/user-guide.md](docs/user-guide.md) and [docs/spec.md](docs/spec.md).

Current MCP tools:

- `memory_remember`
- `memory_search`
- `memory_read_active_collaboration_policy`
- `memory_fetch`
- `memory_recent`
- `memory_feedback`
- `memory_stats`

`memory_search` keeps the default response compact. When debugging retrieval, callers can opt in to per-result diagnostics with `include_diagnostics=true`.

## Runtime Notes

Breathing Memory stores data under the user app-data directory resolved by `platformdirs`, then separates memory by project identity. The exact active path, retrieval mode, and Codex registration state can be inspected with `breathing-memory doctor`.

For Codex installs, `install-codex` pins the MCP registration to a stable project identity derived from the repository at install time, so the live MCP server does not drift with editor-side working directories. Older unpinned databases are not auto-migrated.

Runtime setup is intentionally framed as `super_lite` and `default`. Semantic warmup, HNSW readiness, and other operational details are documented in [docs/user-guide.md](docs/user-guide.md). Normative runtime behavior lives in [docs/spec.md](docs/spec.md).

## Documentation Map

| Document | Primary role | Read this when |
| --- | --- | --- |
| `README.md` | product entrypoint and public surface overview | you need the fastest overview of installation paths, usage surfaces, and the current supported clients |
| `docs/user-guide.md` | user-facing operation details | you are installing, registering, inspecting, or operating Breathing Memory as a runtime |
| `docs/dev-guide.md` | contributor-oriented setup and codebase orientation | you are editing this repository or integrating against the typed core API during development |
| `docs/spec.md` | normative behavior and implementation contract | you need the source of truth for runtime semantics, storage rules, and public behavior |
| `docs/design-rationale.md` | adopted design decisions and tradeoffs | you need to understand why the current boundaries and policies exist |

## Further Reading

- [docs/user-guide.md](docs/user-guide.md): installation, runtime operation, storage behavior, and MCP tool usage
- [docs/dev-guide.md](docs/dev-guide.md): contributor-oriented setup and repository layout
- [docs/spec.md](docs/spec.md): normative behavior and implementation-facing rules
- [docs/design-rationale.md](docs/design-rationale.md): adopted design choices and the reasons behind them
