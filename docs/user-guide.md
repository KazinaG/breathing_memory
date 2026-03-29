# Breathing Memory User Guide

This guide covers user-facing operation details that are too specific for the top-level README and too operational for the normative specification.

It is not the behavioral source of truth. Normative rules live in [spec.md](spec.md).

## Installation Paths

The intended long-term user path is:

```bash
pip install breathing-memory
breathing-memory install-codex
```

For development work or unreleased changes, use one of these paths instead:

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
- which retrieval mode is configured and which mode will actually be used at runtime
- whether the optional semantic extra is available in the current Python environment
- whether the expected Codex MCP registration already exists
- which next action is recommended from the current state

In container-like environments, `doctor` also warns when the default app-data root does not appear to be a dedicated mount, because memory may not survive container rebuilds in that setup.
If `breathing-memory[semantic]` is installed, `doctor` will show that `auto` resolves to `lite`; otherwise it will show that `auto` resolves to `super_lite`.

`breathing-memory install-codex` registers the user-scoped MCP entry named `breathing-memory`, pins that Codex registration to a stable project identity for the current repository, and creates or updates the managed Breathing Memory block in the current repository's `AGENTS.md`.
After a successful run, it also performs a lightweight registration post-check and prints the active project identity, the resolved DB path, the effective retrieval mode, and the next verification step.

Semantic quick check:

```bash
pip install 'breathing-memory[semantic]'
breathing-memory doctor
```

After that, `doctor` should report:

- `configured_mode: auto`
- `effective_mode: lite` or `effective_mode: default`
- `semantic_extra_available: true`

Registration notes:

- the command is idempotent when the existing registration already matches `breathing-memory serve`
- the command stores a stable `BREATHING_MEMORY_PROJECT_ID` in the Codex MCP registration unless you explicitly override it with `BREATHING_MEMORY_PROJECT_ID` or `BREATHING_MEMORY_DB_PATH`
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

For Codex installs, `install-codex` pins the MCP registration to a stable project identity derived from the current repository at install time. That means the live MCP server no longer depends on VSCode or Codex internal working directories when choosing its DB path.

The exact active path is easiest to inspect through:

```bash
breathing-memory doctor
```

When a pinned Codex registration exists, `doctor` uses that registration-derived project identity as its primary diagnostic target. If no Codex registration exists, `doctor` falls back to its own current working directory as before.

If you want to keep memory from an older unpinned Codex registration, move the existing `memory.sqlite3` manually into the new pinned location. Breathing Memory intentionally does not auto-migrate or auto-merge databases.

## Environment Variables

- `BREATHING_MEMORY_DB_PATH`: explicit SQLite database path override
- `BREATHING_MEMORY_PROJECT_ID`: explicit project identity override for storage isolation
- `BREATHING_MEMORY_TOTAL_CAPACITY_MB`: advanced override for total remembered-fragment capacity

`BREATHING_MEMORY_DB_PATH` has the highest priority. If it is set, project auto-resolution is skipped.

For normal use, prefer the CLI option `breathing-memory install-codex --total-capacity-mb ...` over setting `BREATHING_MEMORY_TOTAL_CAPACITY_MB` directly. The environment variable exists as an advanced override for testing, constrained environments, and debugging.

## MCP Tool Surface

Breathing Memory currently exposes six MCP tools:

- `memory_remember`
- `memory_search`
- `memory_fetch`
- `memory_recent`
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

Use `memory_recent` as a caller-side first check before `memory_remember` when you suspect an immediately repeated save.
When `reply_to` is present, `memory_remember` suppresses duplicate deferred `agent` capture for the same `reply_to` and content.
For `user` messages, treat duplicate checks as caller-side logic and use `memory_recent` before `memory_remember`.

Use `source_fragment_ids` only when the deferred final answer materially used remembered fragments. `memory_search` itself does not record references.
Track those materially used fragment ids while drafting the answer so they can be carried into the deferred `memory_remember(actor="agent")` call on the next user turn.

### `memory_search`

Inputs:

- `query`
  - choose the query for the current user request
  - keep the query in the user's language and avoid unnecessary translation
  - rewrite it into a search-oriented query when that improves retrieval
- optional `result_count`
- optional `search_effort`
- optional `include_diagnostics`
  - when `true`, each result includes mode-specific retrieval diagnostics such as lexical rank details or semantic similarity

By default, the packaged runtime uses `super_lite`, which performs lexical retrieval only.

If the optional semantic extra is installed:

```bash
pip install 'breathing-memory[semantic]'
```

then runtime `auto` can resolve to:

- `default` when the embedding backend and a healthy HNSW index are available
- `lite` when embeddings are available but the HNSW index is missing, invalid, or rebuilding
- `super_lite` when semantic retrieval is unavailable

`breathing-memory doctor` reports the configured retrieval mode, the effective runtime mode, and whether the HNSW index is ready.

### `memory_fetch`

Inputs:

- `fragment_id` or `anchor_id`

`memory_fetch` performs direct lookup rather than relevance ranking. `fragment_id` returns that fragment itself. `anchor_id` returns the fragments under that semantic item, ordered by descending `search_priority`.

### `memory_recent`

Inputs:

- optional `limit`
- optional `actor`
- optional `reply_to`

Use `memory_recent` to inspect the latest remembered root fragments before calling `memory_remember` when you need a caller-side duplicate check.

### `memory_feedback`

Use `memory_feedback` when the user clearly confirms, corrects, or evaluates remembered information and the calling agent can attribute that feedback safely.
The calling agent decides whether that feedback applies to the immediately previous answer fragment, to remembered fragments used by that answer, or to both. If attribution is ambiguous, skip `memory_feedback`.

### `memory_stats`

Use `memory_stats` for a compact view of current counts, active parameters, and maintenance state.
