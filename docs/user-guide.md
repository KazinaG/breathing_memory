# Breathing Memory User Guide

This guide covers user-facing operation details that are too specific for the top-level README and too operational for the normative specification.

It is not the behavioral source of truth. Normative rules live in [spec.md](spec.md).

## Setup Positioning

Use [README.md](../README.md) for the shortest installation and quickstart path.
`uv tool install` is the primary user installation path for the packaged CLI.

This guide starts after that point and focuses on:

- how to verify the active runtime state
- how Codex registration behaves
- how the in-process core API differs from the MCP path
- what operational knobs and runtime surfaces exist

For development work or unreleased changes, use [dev-guide.md](dev-guide.md).

## Registration And First Checks

The fastest first check is:

```bash
breathing-memory doctor
```

`doctor` reports:

- whether `breathing-memory` is on `PATH`
- whether `codex` is on `PATH`
- whether the current runtime looks like a container or DevContainer
- where the active Codex registration was found
- whether the active registration uses a PATH command or an absolute executable path
- which project identity is active
- whether the default app-data root is writable
- which SQLite path will be used
- why that SQLite path was selected
- which retrieval mode is configured and which mode will actually be used at runtime
- whether the optional semantic extra is available in the current Python environment
- whether the expected Codex MCP registration already exists
- whether the managed Breathing Memory block in `AGENTS.md` looks current for the installed package
- which next action is recommended from the current state

In container-like environments, `doctor` also warns when the default app-data root does not appear to be a dedicated mount, because memory may not survive container rebuilds in that setup.
If `breathing-memory[semantic]` is installed and HNSW support is available, `doctor` will show that `auto` targets `default` even when the index still needs rebuild or repair; otherwise it will show `super_lite` or the internal `lite` fallback based on what the runtime can actually use.
If you installed the CLI with `uv tool install ...` and `breathing-memory` is still not found on `PATH`, run `uv tool update-shell`, open a new shell, and retry `breathing-memory doctor`.

`breathing-memory install-codex` registers the user-scoped MCP entry named `breathing-memory`, pins that Codex registration to a stable project identity for the current repository, and creates or updates the managed Breathing Memory block in the current repository's `AGENTS.md`.
After a successful run, it also performs a lightweight registration post-check and prints the active project identity, the resolved DB path, the effective retrieval mode, and the next verification step.
By default it writes to the user-level Codex config. If you want repository-local Codex config instead, choose it explicitly with `--codex-config repo`.
Before it writes the Codex registration, it also checks whether the default Breathing Memory app-data root is writable. If that preflight fails, set `BREATHING_MEMORY_DB_PATH` to a writable SQLite path and rerun the command.
After upgrading the package, rerun `breathing-memory install-codex` to refresh the managed Breathing Memory block in `AGENTS.md`.
If you forget, `breathing-memory doctor` will warn when the managed `AGENTS.md` guidance is stale for the installed package.

Codex registration targets:

- `breathing-memory install-codex`: write to the user-level Codex config
- `breathing-memory install-codex --codex-config repo`: write to `.codex/config.toml` in the current repository

After installing `breathing-memory[semantic]`, `doctor` should report:

- `configured_mode: auto`
- `effective_mode: default` when HNSW support is available
- `effective_mode: lite` only as an internal fallback when semantic support is partial
- `semantic_extra_available: true`

Registration notes:

- the command is idempotent when the existing registration already matches `breathing-memory serve`
- the command stores a stable `BREATHING_MEMORY_PROJECT_ID` in the Codex MCP registration unless you explicitly override it with `BREATHING_MEMORY_PROJECT_ID` or `BREATHING_MEMORY_DB_PATH`
- if a different registration already exists under the same name, the command fails and tells you to remove it with `codex mcp remove breathing-memory`
- if `codex` is not on `PATH`, the command exits clearly instead of writing partial state
- if writing the default user-level Codex config fails, the command suggests `--codex-config repo` as the next step
- if `AGENTS.md` cannot be created or safely updated, the command fails instead of leaving repo-side workflow in a partial state

## Runtime Commands

Server entrypoints:

- `python -m breathing_memory`
- `python -m breathing_memory serve`
- `breathing-memory serve`

All three start the stdio MCP server.
The `serve` command keeps the MCP handshake path light, then starts a best-effort background semantic warmup only after the MCP session is live. That warmup is an optimization, not a new contract: the first semantic call may still wait, and semantic calls can still fail if import, download, or model initialization fails.

Inspection commands:

- `breathing-memory doctor`: installation and environment checks
- `breathing-memory inspect-memory`: compact inspection output
- `breathing-memory inspect-memory --json`: machine-readable memory state
- `breathing-memory warmup`: eagerly load the semantic embedding backend for the current environment

If you installed the CLI with `uv tool install ...`, upgrade it with:

```bash
uv tool upgrade breathing-memory
breathing-memory install-codex
```

To remove a `uv tool` install:

```bash
uv tool uninstall breathing-memory
```

For first-time installation, the matching flow is:

```bash
uv tool install 'breathing-memory[semantic]'
breathing-memory doctor
breathing-memory install-codex
```

The `semantic` extra is the recommended install target, but it is much larger than the minimal lexical-only package because it pulls local embedding dependencies. If you want the smallest initial install, use `uv tool install breathing-memory` and accept `super_lite` retrieval.

## In-Process Core API

Breathing Memory keeps the MCP surface stable, but it is not MCP-only. Non-MCP consumers can use the typed core service directly through `breathing_memory.core`.

Use the public factory when you want the same default project-scoped config resolution that the CLI uses:

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
engine.read_active_collaboration_policy(ReadActiveCollaborationPolicyRequest())
engine.close()
```

If you need to inspect or override the resolved runtime config before constructing the engine, use `breathing_memory.core.resolve_memory_config(...)` and pass that `MemoryConfig` into `create_engine(...)`.

Use `breathing_memory.engine.BreathingMemoryEngine` only when dict-shaped compatibility is still required. That module is a legacy shim over the typed core service.

Use this path when your runtime is already in Python and does not need MCP tool transport. Use the MCP path when your runtime expects stdio tools and schema-validated inputs.

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
- `BREATHING_MEMORY_ACP_TOKEN_BUDGET`: advanced override for the default collaboration-policy token budget when the caller omits `token_budget`
- `BREATHING_MEMORY_MCP_PAYLOAD_MODE`: advanced override for MCP payload shape, `slim` or `debug`

`BREATHING_MEMORY_DB_PATH` has the highest priority. If it is set, project auto-resolution is skipped.

For normal use, prefer the CLI option `breathing-memory install-codex --total-capacity-mb ...` over setting `BREATHING_MEMORY_TOTAL_CAPACITY_MB` directly. The environment variable exists as an advanced override for testing, constrained environments, and debugging.

## MCP Tool Surface

Breathing Memory currently exposes seven MCP tools:

- `memory_remember`
- `memory_search`
- `memory_read_active_collaboration_policy`
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
- `kind`

Conversation capture timing:

- save each user message when that user turn arrives
- save the immediately previous final agent answer on the next user turn
- if no later user turn arrives, the final agent answer may remain unremembered

Use `memory_recent` as a caller-side first check before `memory_remember` when you suspect an immediately repeated save.
When `reply_to` is present, `memory_remember` suppresses duplicate deferred `agent` capture for the same `reply_to` and content.
For `user` messages, treat duplicate checks as caller-side logic and use `memory_recent` before `memory_remember`.

Use `source_fragment_ids` only when the deferred final answer materially used remembered fragments. `memory_search` itself does not record references.
Track those materially used fragment ids while drafting the answer so they can be carried into the deferred `memory_remember(actor="agent")` call on the next user turn.

`kind="collaboration_policy"` is reserved for derived agent-side collaboration guidance.
Callers may save these fragments when a reusable rule about how to collaborate with the user becomes clear, whether that comes from explicit user feedback or from broader conversational context.
Prefer saving them only when they are likely to affect future behavior, choices, or response style.
Do not save weak inferences, one-off requests, transient emotions, or ambiguous signals as collaboration policy.
When uncertain, prefer not to save.

### `memory_search`

Inputs:

- `query`
  - choose the query for the current user request
  - keep the query in the user's language and avoid unnecessary translation
  - rewrite it into a search-oriented query when that improves retrieval
- optional `result_count`
  - defaults to `4`; accepted values are `4 * 2^n`
- optional `search_effort`
- optional `include_diagnostics`
  - when `true`, each result includes mode-specific retrieval diagnostics such as lexical rank details or semantic similarity

By default, the packaged runtime uses `super_lite`, which performs lexical retrieval only.

If the optional semantic extra is installed:

```bash
uv tool install 'breathing-memory[semantic]'
```

then runtime `auto` can resolve to:

- `default` when the embedding backend and HNSW support are available
- `super_lite` when semantic retrieval is unavailable

If embeddings are available but HNSW support is unavailable, runtime currently falls back to `lite` internally. That state is still surfaced by diagnostics, but it is not treated as a primary installation target in this guide.

`breathing-memory doctor` reports the configured retrieval mode, the effective runtime mode, and whether the HNSW index is ready.
`breathing-memory warmup` is re-runnable. Use it when you want to preload the semantic backend explicitly or retry warmup after a transient failure without starting a full MCP session.
When semantic retrieval encounters live fragments with missing embeddings, Breathing Memory backfills those vectors before continuing. If `default` search needs ANN rebuild or repair work, it tries that first, waits briefly for conflicting maintenance, and may return a retryable or non-retryable status so the caller can decide what to do next.

For `default` retrieval in slim containers, `hnswlib` may need system packages that are not present by default. Typical requirements are:

- Python headers for the active interpreter, such as `python3-dev` or `python3.12-dev`
- compiler and build tools, such as `build-essential` on Debian/Ubuntu

If `sentence-transformers` installs but `hnswlib` does not, `doctor` will usually show semantic support as available while runtime falls back to the internal `lite` state instead of `default`.

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
