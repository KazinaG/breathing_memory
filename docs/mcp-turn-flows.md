# MCP Turn Flows

This document focuses on the active dedicated-API design target for Breathing Memory turn start.

It is not the normative source of truth. Current required behavior still lives in [spec.md](spec.md) and in the managed Breathing Memory block that `breathing-memory install-codex` writes into `AGENTS.md`.

## Active Design Target: Dedicated `turn_start_state` API

### Design Goals

- Minimize Codex decision turns first.
- Keep the context needed for each turn as small as possible.
- Treat `Read repository AGENTS.md` as a repository workflow gate, not as a memory-protocol RTT phase.
- Use a dedicated turn-start API instead of overloading `memory_recent(...)` with thread resolution.
- Keep `memory_recent(...)` as a low-level recent-fragment tool unless and until callers no longer need it.

### API Direction

This design does not treat the new API as a successor to `memory_recent(...)`.

`memory_recent(...)` remains a low-level tool for checking recent remembered fragments.

The active design target is a dedicated turn-start API, tentatively named `turn_start_state(...)`, that returns the state needed to close the turn-start execution plan.

The expected response shape is:

- `previous_agent_source_user`
  The single user fragment that the expected previous agent answer replied to, when that source user exists.
- `previous_agent_answer_state`
  Whether the expected previous agent answer is already remembered and, if so, which existing anchor should be reused.
- `current_user_message_state`
  Whether the current user message is already remembered and, if so, which existing anchor should be reused.
- `resolved_reply_to`
  The reply target that should be used if the current user message must be saved.

### Sequence View

This view shows who calls what, where waiting happens, and why the common case should collapse to two Codex decision turns.

```mermaid
sequenceDiagram
    participant Codex
    participant BM as Breathing Memory MCP

    Note over Codex: Repository workflow gate<br/>Read repository AGENTS.md

    par Phase 1: load turn-start state
        Codex->>BM: turn_start_state(...)
    and
        Codex->>BM: memory_read_active_collaboration_policy()
    end

    BM-->>Codex: previous_agent_source_user
    BM-->>Codex: previous_agent_answer_state
    BM-->>Codex: current_user_message_state
    BM-->>Codex: resolved_reply_to
    BM-->>Codex: ACP payload

    Note over Codex: Phase 1: build execution plan<br/>from dedicated turn-start state

    alt previous final agent must be saved
        Note over Codex: Phase 2: conditional mutations
        Codex->>BM: memory_remember(actor="agent")
        BM-->>Codex: previous-agent anchor
    else reuse previous-agent anchor
        Note over Codex: Phase 2: conditional mutations
        Note over Codex: Reuse previous-agent anchor
    end

    alt current user must be saved
        Codex->>BM: memory_remember(actor="user", reply_to=resolved_reply_to)
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

- The target common case is one phase-1 state load followed by phase-2 mutations.
- The dedicated API should tell Codex whether the previous final agent answer already exists, whether the current user message already exists, and what `reply_to` should be used if the user message must be saved.
- `previous_agent_source_user` is expected to be a single object, not a candidate list.
- ACP remains a separate read API even in this design.

### Flowchart View

This view shows the same design as phases and dependency barriers rather than actors and messages.

```mermaid
flowchart TD
    A[Read repository AGENTS.md]

    subgraph P1[Phase 1: load turn-start state and build execution plan]
        P1I([Enter Phase 1])
        B[Call turn_start_state]
        C[Read active collaboration policy]
        W1{{Wait for phase-1 state}}
        D[Build execution plan from dedicated state]
        E{Execution plan ready?}
        P1I --> B
        P1I --> C
        B --> W1
        C --> W1
        W1 --> D
        D --> E
    end

    subgraph P2[Phase 2: conditional mutations]
        P2I([Enter Phase 2])
        F{Need to save previous final agent?}
        G[Reuse previous-agent anchor]
        H[memory_remember agent]
        W2{{Wait for previous-agent anchor state}}
        I{Need to save current user?}
        J[Reuse existing user anchor]
        K[memory_remember user with resolved_reply_to]
        W3{{Wait for current-user state}}
        P2I --> F
        F -- No --> G
        F -- Yes --> H
        G --> W2
        H --> W2
        W2 --> I
        I -- No --> J
        I -- Yes --> K
        J --> W3
        K --> W3
    end

    A --> P1I
    E -- No --> B
    E -- Yes --> P2I
    W3 --> L{Need more retrieval for this answer?}
    L -- Yes --> M[memory_search]
    L -- No --> N[Continue normal work]
    M --> L
```

Notes:

- The design target is to make phase 1 deterministic enough that retry inside phase 1 is exceptional rather than normal.
- `turn_start_state(...)` should return exactly the state needed for reply-target safety and duplicate avoidance, not a generic recent-fragment list.
- `memory_recent(...)` is no longer the active extension point in this design; it stays as a lower-level API.
- The wait nodes show the real dependency barriers: after turn-start state load, after previous-agent anchor state, and after current-user state.
- Current-user mutation still waits on `resolved_reply_to` stability. When the previous agent must be newly saved, phase 2 remains partially serial for that branch.

## Open Questions

- What should the exact request shape of `turn_start_state(...)` be?
- Should `previous_agent_answer_state` and `current_user_message_state` return only `existing_anchor_id | needs_save`, or should they also include the matched fragment content?
- Should `memory_recent(...)` remain permanently as a low-level helper, or should it be deprecated after callers migrate to the dedicated turn-start API?
