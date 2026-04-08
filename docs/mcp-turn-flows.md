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

    par Phase 0: candidate gathering
        Codex->>BM: memory_recent(agent candidates)
    and
        Codex->>BM: memory_recent(user candidates)
    and
        Codex->>BM: memory_read_active_collaboration_policy()
    end

    BM-->>Codex: recent agent candidates
    BM-->>Codex: recent user candidates
    BM-->>Codex: ACP payload

    Note over Codex: Phase 1: build execution plan<br/>resolve anchor threading<br/>decide duplicate handling

    alt previous final agent must be saved
        Codex->>BM: memory_remember(actor="agent")
        BM-->>Codex: previous-agent anchor
    else reuse previous-agent anchor
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

- The target common case is `gather -> mutate`, which keeps Codex to two decision turns.
- The branch case becomes `gather -> remember agent -> remember user` when the current user save depends on a newly created previous-agent anchor.
- ACP is loaded during phase 0 so it is ready before retrieval planning and normal work continue.

### Flowchart View

This view shows the same design as phases and dependency barriers rather than actors and messages.

```mermaid
flowchart TD
    A[Read repository AGENTS.md]

    subgraph P0[Phase 0: parallel candidate gathering]
        B[memory_recent agent candidates]
        C[memory_recent user candidates]
        D[memory_read_active_collaboration_policy]
        W0{{Wait for phase-0 results}}
        B --> W0
        C --> W0
        D --> W0
    end

    subgraph P1[Phase 1: build execution plan]
        E[Resolve reply target and duplicate handling from gathered candidates]
        F{Need to save previous final agent?}
        G[Reuse previous-agent anchor]
        H[memory_remember agent]
        W1{{Wait for previous-agent anchor state}}
        E --> F
        F -- No --> G
        F -- Yes --> H
        G --> W1
        H --> W1
    end

    subgraph P2[Phase 2: conditional user mutation]
        I{Need to save current user?}
        J[Reuse existing user anchor]
        K[memory_remember user]
        W2{{Wait for current-user state}}
        I -- No --> J
        I -- Yes --> K
        J --> W2
        K --> W2
    end

    A --> B
    A --> C
    A --> D
    W0 --> E
    W1 --> I
    W2 --> L{Need more retrieval for this answer?}
    L -- Yes --> M[memory_search]
    L -- No --> N[Continue normal work]
    M --> L
```

Notes:

- The design target is to make phase-0 candidate gathering rich enough that phase 1 can decide both threading and duplicate handling without another read in the common case.
- `memory_recent(agent)` should gather enough signal for previous-agent duplicate detection and anchor-resolution planning.
- `memory_recent(user)` should gather user-side candidates, not just a single duplicate check result.
- The wait nodes show the real dependency barriers: after candidate gathering, after previous-agent anchor state, and after current-user state.

## Open Questions

- What is the minimum `memory_recent(user)` payload that still lets phase 1 decide current-user save behavior without an extra read in the common case?
- What recent-agent fields are sufficient for previous-agent anchor resolution while keeping phase-0 context small?
- If phase-0 candidates are insufficient, what is the cleanest fallback without losing the main goal of minimizing Codex decision turns?
