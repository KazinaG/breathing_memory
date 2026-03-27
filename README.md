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

`breathing-memory install-codex` registers the `breathing-memory` MCP server with the currently supported client and creates or updates the managed Breathing Memory block in the current repository's `AGENTS.md`.

Published package:

- `pip install breathing-memory`

Development installs:

```bash
pip install git+https://github.com/KazinaG/breathing_memory.git
# or inside a clone:
pip install -e .
breathing-memory install-codex
```

Release notes:

- PyPI publish runs from `.github/workflows/publish.yml`
- `v0.1.0` is published on PyPI
- pushing a tag such as `v0.1.1` triggers the build and PyPI publish workflow

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

- `breathing-memory doctor`: inspect installation, active project identity, DB path selection, and client registration state
- `breathing-memory serve`: start the stdio MCP server
- `breathing-memory inspect-memory --json`: inspect current memory state

## How Memory Works

Breathing Memory does not auto-capture the full client conversation by itself. The supported operating path is explicit MCP use by the calling agent.

The basic flow is:

1. If there is an unremembered final agent answer from the previous turn, save it first with `memory_remember(actor="agent")`
2. Save the current user message with `memory_remember(actor="user")`
3. Search before an answer with `memory_search`
4. Record feedback with `memory_feedback` when the user clearly confirms or corrects remembered information

Key points:

- one user utterance becomes one fragment
- one final user-facing agent answer is normally remembered on the next user turn
- commentary is not remembered
- if the final answer materially used remembered fragments, pass those ids in `source_fragment_ids`
- edits are modeled as forks rather than overwrites
- duplicate deferred agent capture for the same reply target and content is suppressed
- archived runtime files such as `archived_sessions/*.jsonl` are not the primary capture path
- if no later user turn arrives, the final agent answer may remain unremembered

Current MCP tools:

- `memory_remember`
- `memory_search`
- `memory_fetch`
- `memory_feedback`
- `memory_stats`

## Runtime Notes

Breathing Memory stores data under the user app-data directory resolved by `platformdirs`, then separates memory by project identity. The exact SQLite path can be inspected with `breathing-memory doctor`.

The current implementation is text-only. Runtime `auto` resolves to `super_lite`, which performs lexical retrieval only. The public search surface is already aligned for later semantic retrieval work, but explicit `lite` and `default` modes are not supported in this slice.

The current compression backend invokes a supported coding agent without leaving normal conversation history. In the current supported setup, that path uses Codex through `codex exec --ephemeral`.

## Further Reading

- [docs/user-guide.md](docs/user-guide.md): installation, runtime operation, storage behavior, and MCP tool usage
- [docs/dev-guide.md](docs/dev-guide.md): contributor-oriented setup and repository layout
- [docs/spec.md](docs/spec.md): normative behavior and implementation-facing rules
- [docs/design-rationale.md](docs/design-rationale.md): adopted design choices and the reasons behind them
