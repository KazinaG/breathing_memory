# Breathing Memory Design Rationale

This document records why Breathing Memory adopts its current design choices.

It is not the normative specification. Behavioral source of truth lives in [spec.md](spec.md).

## Memory Boundaries

### Why memory is not repository source of truth

Breathing Memory is for high-churn collaboration context that an agent should remember but a repository should not need to constantly document. Product rules, stable protocols, and durable project knowledge belong in repository documents. Conversational habits, temporary preferences, and ephemeral working context belong here.

### Why storage lives under user app-data but remains project-scoped

The runtime behaves like a normal user-level application, so its default SQLite state belongs under user app-data rather than inside each consuming repository. That default also fits the kind of data Breathing Memory stores: conversational history, personal working habits, temporary preferences, and other user-level memory that should follow the user rather than live in version-controlled project files. At the same time, remembered content is isolated by project identity so unrelated repositories and directories do not share one memory space by accident.

### Why capture is explicit MCP rather than runtime-log import

Explicit MCP calls make remembered input intentional and inspectable. Codex runtime artifacts such as archived JSONL files are implementation details and are too unstable to serve as the primary operating path.

## Representation Choices

### Why `anchor` and `fragment` are separate

An `anchor` represents the semantic conversation item being answered. A `fragment` is one concrete realization of that item. The split is needed because compression can create additional child fragments while reply structure and source-side continuity still belong to one semantic node.

### Why `fragments` are live-only

Forgotten fragments are physically purged instead of being kept as soft-deleted rows. This keeps the concrete fragment table simple and current. Any maintenance history that still matters after purge is summarized into aggregates such as `sequence_metrics`.

### Why edits are modeled as forks

Editing an earlier Codex message is treated as creating a new branch, not overwriting the old one. Overwrite semantics would rewrite past references, feedback, and compression outcomes. A fork preserves append-order history and keeps prior branches inspectable.

## Retrieval And Attribution

### Why references only count material use

Search hits alone are predictions. Breathing Memory records a reference only when the final answer actually uses a remembered fragment. This keeps reference growth tied to concrete use instead of to broad retrieval noise.

### Why feedback attribution is delegated to the agent

User replies often evaluate either the immediately preceding answer, one of the remembered fragments used in that answer, or both. A rigid fixed rule would misattribute many cases. Letting Codex attribute feedback keeps the system aligned with future model improvements.

### Why observation wins over retrieval prediction

Retrieval ranking is only a prediction about what should matter. A concrete final-answer reference is an observation that a fragment did matter. When the two disagree, the observed use is treated as the stronger signal.

## Retrieval Architecture

### Why semantic retrieval uses a pluggable embedding backend

Embedding generation is intentionally separated from storage, indexing, and reranking. This keeps the backend boundary narrow, allows model replacement, and prevents the public retrieval API from depending on one mandatory provider.
