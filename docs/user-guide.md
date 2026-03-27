# Breathing Memory User Guide

This guide covers user-facing operation details that are too specific for the top-level README and too operational for the normative specification.

It is not the behavioral source of truth. Normative rules live in [spec.md](spec.md).

## Installation Paths

The intended long-term user path is:

```bash
pip install breathing-memory
breathing-memory install-codex
```

Published-package verification is still pending. Until that is available, use one of these paths:

### Install From Git

```bash
pip install git+https://github.com/KazinaG/breathing_memory.git
breathing-memory install-codex
```

### Install From A Local Clone

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
breathing-memory install-codex
```

## Registration And First Checks

The fastest first check is:

```bash
breathing-memory doctor
```

`doctor` reports:

- whether `breathing-memory` is on `PATH`
- whether `codex` is on `PATH`
- whether the current runtime looks like a container or DevContainer
- which project identity is active
- which SQLite path will be used
- whether the expected Codex MCP registration already exists
- which next action is recommended from the current state

In container-like environments, `doctor` also warns when the default app-data root does not appear to be a dedicated mount, because memory may not survive container rebuilds in that setup.

`breathing-memory install-codex` registers the user-scoped MCP entry named `breathing-memory` and creates or updates the managed Breathing Memory block in the current repository's `AGENTS.md`.
After a successful run, it also performs a lightweight registration post-check and prints the active project identity, the resolved DB path, and the next verification step.

Registration notes:

- the command is idempotent when the existing registration already matches `breathing-memory serve`
- if a different registration already exists under the same name, the command fails and tells you to remove it with `codex mcp remove breathing-memory`
- if `codex` is not on `PATH`, the command exits clearly instead of writing partial state
- if `AGENTS.md` cannot be created or safely updated, the command fails instead of leaving repo-side workflow in a partial state

## Runtime Commands

Server entrypoints:

- `python -m breathing_memory`
- `python -m breathing_memory serve`
- `breathing-memory serve`

All three start the stdio MCP server.

Inspection commands:

- `breathing-memory doctor`: installation and environment checks
- `breathing-memory inspect-memory`: compact inspection output
- `breathing-memory inspect-memory --json`: machine-readable memory state

## Storage Behavior

Breathing Memory stores data under the user app-data directory resolved by `platformdirs`, then separates memory by project identity.

Typical roots:

- Linux: `~/.local/share/breathing-memory/projects/...`
- macOS: `~/Library/Application Support/breathing-memory/projects/...`
- Windows: `%LOCALAPPDATA%\\OpenAI\\breathing-memory\\projects\\...`

Project identity is resolved in this order:

1. `BREATHING_MEMORY_PROJECT_ID`
2. git repository root of the current working directory
3. current working directory realpath

Each identity is mapped to a stable project key, and the SQLite file lives at:

```text
<app-data>/breathing-memory/projects/<project-key>/memory.sqlite3
```

This keeps memory out of the consuming repository while preventing unrelated repositories from sharing the same database by default.

The exact active path is easiest to inspect through:

```bash
breathing-memory doctor
```

## Environment Variables

- `BREATHING_MEMORY_DB_PATH`: explicit SQLite database path override
- `BREATHING_MEMORY_PROJECT_ID`: explicit project identity override for storage isolation

`BREATHING_MEMORY_DB_PATH` has the highest priority. If it is set, project auto-resolution is skipped.

User-facing settings such as total capacity live in `MemoryConfig` in [config.py](../src/breathing_memory/config.py). Environment variables are intentionally kept minimal and focused on storage-path overrides.

## MCP Tool Surface

Breathing Memory currently exposes five MCP tools:

- `memory_remember`
- `memory_search`
- `memory_fetch`
- `memory_feedback`
- `memory_stats`

### `memory_remember`

Required inputs:

- `content`
- `actor`

Optional inputs:

- `reply_to`
- `source_fragment_ids`

Conversation capture timing:

- save each user message when that user turn arrives
- save the immediately previous final agent answer on the next user turn
- if no later user turn arrives, the final agent answer may remain unremembered

Use `source_fragment_ids` only when the deferred final answer materially used remembered fragments. `memory_search` itself does not record references.

### `memory_search`

Inputs:

- `query`
  - keep the query in the user's language and avoid unnecessary translation or paraphrase
- optional `result_count`
- optional `search_effort`

The current implementation is text-only. Runtime `auto` resolves to `super_lite`, which performs lexical retrieval only. The public search surface is already aligned for later semantic retrieval work, but explicit `lite` and `default` modes are not yet supported in this slice.

### `memory_fetch`

Inputs:

- `fragment_id` or `anchor_id`

`memory_fetch` performs direct lookup rather than relevance ranking. `fragment_id` returns that fragment itself. `anchor_id` returns the fragments under that semantic item, ordered by descending `search_priority`.

### `memory_feedback`

Use `memory_feedback` when the user clearly confirms, corrects, or evaluates remembered information and the calling agent can attribute that feedback safely.

### `memory_stats`

Use `memory_stats` for a compact view of current counts, active parameters, and maintenance state.
