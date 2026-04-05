# Breathing Memory Development Guide

This guide collects contributor-oriented setup details that do not belong in the top-level README.

## Local Setup

Inside this repository:

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[semantic]'
python -m unittest discover -s tests
python -m breathing_memory
```

`python -m breathing_memory` and `python -m breathing_memory serve` both start the server.
If you prefer the traditional tooling flow, `python -m venv` plus `pip install -e '.[semantic]'` remains valid.

## Repository Layout

- `src/breathing_memory/`: package source
- `src/breathing_memory/core/`: core service layer intended for in-process reuse
- `src/breathing_memory/core/ports.py`: protocol boundary for store, embedding, ANN, and compression dependencies
- `src/breathing_memory/adapters/`: transport-facing adapters and payload serializers such as MCP
- `tests/`: unit tests
- `docs/spec.md`: implementation-facing specification
- `docs/design-rationale.md`: adopted design choices and their rationale
- `docs/user-guide.md`: user-facing operational guide between the README and the specification

## Core API

Prefer the typed core API for in-process consumers:

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
```

When a consumer needs to inspect or override the resolved runtime location without rebuilding the rest of the wiring, use `breathing_memory.core.resolve_memory_config(...)` and pass the result back to `create_engine(...)`.

Use `breathing_memory.engine.BreathingMemoryEngine` only when dict-shaped compatibility is still required. That module is a legacy shim over the typed core service.

## Embedding Into Another Runtime

If you are integrating Breathing Memory into another Python system, prefer the typed core API over the MCP transport.

Use the MCP server when:

- your host runtime already expects stdio tools
- you want schema-validated tool calls as the integration boundary

Use the typed core API when:

- your host runtime already runs in Python
- you want direct request/response objects instead of transport payloads
- you want to share one process with your application or TUI

Recommended starting point:

```python
from breathing_memory.core import (
    ReadActiveCollaborationPolicyRequest,
    RememberRequest,
    SearchRequest,
    create_engine,
)

memory = create_engine()
memory.remember(RememberRequest(content="hello", actor="user"))
hits = memory.search(SearchRequest(query="hello"))
policy = memory.read_active_collaboration_policy(ReadActiveCollaborationPolicyRequest())
```

Treat `breathing_memory.core` as the primary in-process contract. Treat `breathing_memory.engine` as a compatibility path for older dict-shaped callers.
