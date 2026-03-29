# Breathing Memory

Breathing Memory is a local memory support system for coding agents. It runs as a stdio MCP server through the official Python MCP SDK, stores memory in SQLite, and isolates memory by project so one installation can be reused across repositories without mixing contexts.

## Overview

Breathing Memory keeps collaboration context that an agent should remember but a repository should not need to encode everywhere.

- local stdio MCP server
- SQLite storage under user app-data, isolated by project
- fragment-centric public model built around `anchor` and `fragment`
- text-first retrieval today, with a public search surface already aligned for later semantic retrieval work
- dynamic `working / holding` maintenance with a compression backend that uses a supported coding agent without polluting normal conversation history

## Supported Clients

- Codex

## Installation

The intended long-term user path is:

```bash
pip install breathing-memory
breathing-memory install-codex
```

`breathing-memory install-codex` registers the `breathing-memory` MCP server with the currently supported client, pins that registration to a stable project identity for the current repository, and creates or updates the managed Breathing Memory block in the current repository's `AGENTS.md`.

Published package:

- `pip install breathing-memory`
- semantic retrieval: `pip install 'breathing-memory[semantic]'`

Development installs:

```bash
pip install git+https://github.com/KazinaG/breathing_memory.git
# or inside a clone:
pip install -e .
breathing-memory install-codex
```

## Quickstart

Recommended first run:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
breathing-memory doctor
breathing-memory install-codex
```

Useful commands:

- `breathing-memory doctor`: inspect installation, active project identity, DB path selection, Codex registration state, and effective retrieval mode
- `breathing-memory serve`: start the stdio MCP server
- `breathing-memory inspect-memory --json`: inspect current memory state

## How Memory Works

Breathing Memory does not auto-capture the full client conversation by itself. The supported operating path is explicit MCP use by the calling agent.

The basic flow is:

1. Check `memory_recent` before persisting immediately repeated agent / user turns
2. If there is an unremembered final agent answer from the previous turn, save it first with `memory_remember(actor="agent")`
3. Save the current user message with `memory_remember(actor="user")`
4. Search before an answer with `memory_search`
5. Record feedback with `memory_feedback` when the user clearly confirms or corrects remembered information

Key points:

- one user utterance becomes one fragment
- one final user-facing agent answer is normally remembered on the next user turn
- commentary is not remembered
- use `memory_recent` as a caller-side first check before `memory_remember` when you suspect an immediately repeated save
- track which retrieved fragments materially informed the final answer and pass them in `source_fragment_ids`
- if the final answer materially used remembered fragments, pass those ids in `source_fragment_ids`
- use `memory_feedback` only when the user's evaluation can be attributed safely
- edits are modeled as forks rather than overwrites
- duplicate deferred agent capture for the same reply target and content is suppressed
- user duplicate checks are caller-side and should use `memory_recent` rather than engine-side suppression
- archived runtime files such as `archived_sessions/*.jsonl` are not the primary capture path
- if no later user turn arrives, the final agent answer may remain unremembered

Current MCP tools:

- `memory_remember`
- `memory_search`
- `memory_fetch`
- `memory_recent`
- `memory_feedback`
- `memory_stats`

`memory_search` keeps the default response compact. When debugging retrieval, callers can opt in to per-result diagnostics with `include_diagnostics=true`.

## Runtime Notes

Breathing Memory stores data under the user app-data directory resolved by `platformdirs`, then separates memory by project identity. The exact SQLite path can be inspected with `breathing-memory doctor`.

For Codex installs, `install-codex` now pins the MCP registration to a stable project identity derived from the repository at install time, so the live MCP server does not drift with VSCode or Codex internal working directories. `doctor` prefers that registration-derived identity when it is available, so its reported DB path matches the live MCP target rather than the shell's current directory.

If you already have remembered data under an older unpinned Codex registration, migration is manual by design. Move the SQLite database yourself if you want to keep that history; Breathing Memory does not auto-discover or auto-merge old databases.

The current implementation supports lexical retrieval by default and semantic retrieval through the optional `semantic` extra. Runtime `auto` resolves to `default` when the embedding backend and HNSW support are available, resolves to `lite` when embeddings are available but HNSW support is unavailable, and resolves to `super_lite` when semantic retrieval is unavailable. When semantic retrieval encounters live fragments with missing embeddings, Breathing Memory backfills those vectors before continuing. If `default` search finds a missing or invalid ANN index, it attempts repair first, waits briefly for conflicting rebuild work, and returns a structured status when the caller should decide whether to retry or fall back.

`breathing-memory doctor` reports both the configured retrieval mode and the effective runtime mode, along with HNSW support and index readiness, so after installing `breathing-memory[semantic]` you can verify whether `auto` can target the HNSW-backed path and whether the index is already ready or still needs repair.
`breathing-memory install-codex` also prints the effective retrieval mode in its post-install summary, so the semantic state is visible even before the first MCP conversation.

The current compression backend invokes a supported coding agent without leaving normal conversation history. In the current supported setup, that path uses Codex through `codex exec --ephemeral`.

## Further Reading

- [docs/user-guide.md](docs/user-guide.md): installation, runtime operation, storage behavior, and MCP tool usage
- [docs/dev-guide.md](docs/dev-guide.md): contributor-oriented setup and repository layout
- [docs/spec.md](docs/spec.md): normative behavior and implementation-facing rules
- [docs/design-rationale.md](docs/design-rationale.md): adopted design choices and the reasons behind them
