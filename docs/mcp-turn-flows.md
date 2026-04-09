# MCP Turn Flows

This document focuses on the active client-parallel design target for Breathing Memory turn start.

It is not the normative source of truth. Current required behavior still lives in [spec.md](spec.md) and in the managed Breathing Memory block that `breathing-memory install-codex` writes into `AGENTS.md`.

## Active Design Target: Client-Parallel Sketch

### Design Goals

- Minimize Codex decision turns first.
- Keep the context needed for each turn as small as possible.
- Treat `Read repository AGENTS.md` as a repository workflow gate, not as a memory-protocol RTT phase.
- Gather enough candidate memory in the first round so later phases mostly decide and mutate instead of re-reading.

### Sequence View

This view shows who calls what, where waiting happens, and why the common case should collapse to two Codex decision turns.

```mermaid
sequenceDiagram
    participant Codex
    participant BM as Breathing Memory MCP

    Note over Codex: Repository workflow gate<br/>Read repository AGENTS.md

    loop Phase 1-2 until execution plan is stable
        par Phase 1: candidate gathering
            Codex->>BM: memory_recent(agent candidates)
        and
            Codex->>BM: memory_recent(user candidates)
        and
            Codex->>BM: memory_read_active_collaboration_policy()
        end

        BM-->>Codex: recent agent candidates
        BM-->>Codex: recent user candidates
        BM-->>Codex: ACP payload

        Note over Codex: Phase 2: build execution plan<br/>resolve anchor threading<br/>decide duplicate handling
    end

    alt previous final agent must be saved
        Note over Codex: Phase 3: conditional mutations
        Codex->>BM: memory_remember(actor="agent")
        BM-->>Codex: previous-agent anchor
    else reuse previous-agent anchor
        Note over Codex: Phase 3: conditional mutations
        Note over Codex: Reuse previous-agent anchor
    end

    alt current user must be saved
        Codex->>BM: memory_remember(actor="user", reply_to=...)
        BM-->>Codex: user_anchor
    else duplicate user save
        Note over Codex: Reuse existing user anchor
    end

    opt need more retrieval for this answer
        loop until enough context is gathered
            Codex->>BM: memory_search(...)
            BM-->>Codex: relevant fragments
        end
    end

    Note over Codex: Continue normal work
```

Notes:

- The target common case is one pass through phases 1 and 2, followed by phase 3.
- The branch case becomes `gather -> remember agent -> remember user` when the current user save depends on a newly created previous-agent anchor.
- ACP is loaded during phase 1 so it is ready before retrieval planning and normal work continue.
- Current-user mutation is blocked only until its `reply_to` target and duplicate decision are stable. It is not inherently non-parallel, but it cannot safely run before those dependencies are resolved.

### Flowchart View

This view shows the same design as phases and dependency barriers rather than actors and messages.

```mermaid
flowchart TD
    A[Read repository AGENTS.md]

    subgraph P1[Phase 1: parallel candidate gathering]
        B[memory_recent agent candidates]
        C[memory_recent user candidates]
        D[memory_read_active_collaboration_policy]
        W1{{Wait for phase-1 results}}
        B --> W1
        C --> W1
        D --> W1
    end

    subgraph P2[Phase 2: build execution plan]
        E[Resolve reply target and duplicate handling from gathered candidates]
        F{Execution plan stable?}
        E --> F
    end

    subgraph P3[Phase 3: conditional mutations]
        G{Need to save previous final agent?}
        H[Reuse previous-agent anchor]
        I[memory_remember agent]
        W2{{Wait for previous-agent anchor state}}
        J{Need to save current user?}
        K[Reuse existing user anchor]
        L[memory_remember user]
        W3{{Wait for current-user state}}
        G -- No --> H
        G -- Yes --> I
        H --> W2
        I --> W2
        J -- No --> K
        J -- Yes --> L
        K --> W3
        L --> W3
    end

    A --> B
    A --> C
    A --> D
    W1 --> E
    F -- No --> B
    F -- Yes --> G
    W2 --> J
    W3 --> M{Need more retrieval for this answer?}
    M -- Yes --> N[memory_search]
    M -- No --> O[Continue normal work]
    N --> M
```

Notes:

- The design target is to make phase-1 candidate gathering rich enough that phase 2 can decide both threading and duplicate handling without another read in the common case.
- `memory_recent(agent)` should gather enough signal for previous-agent duplicate detection and anchor-resolution planning.
- `memory_recent(user)` should gather user-side candidates, not just a single duplicate check result.
- If phase 2 cannot produce a stable execution plan, the flow loops back to phase 1 for more candidate gathering.
- The wait nodes show the real dependency barriers: after candidate gathering, after previous-agent anchor state, and after current-user state.
- Current-user mutation waits on `reply_to` stability. When `reply_to` depends on a newly created previous-agent anchor, phase 3 becomes partially serial for that branch.

## Open Questions

- What is the minimum `memory_recent(user)` payload that still lets phase 2 decide current-user save behavior without an extra read in the common case?
- What recent-agent fields are sufficient for previous-agent anchor resolution while keeping phase-1 context small?
- If phase-1 candidates are insufficient, what is the cleanest fallback without losing the main goal of minimizing Codex decision turns?
