# Breathing Memory Design Rationale

This document records why Breathing Memory adopts its current design choices.

It is not the normative specification. Behavioral source of truth lives in [spec.md](spec.md).

## Memory Boundaries

### Why memory is not repository source of truth

Breathing Memory is for high-churn collaboration context that an agent should remember but a repository should not need to constantly document. Product rules, stable protocols, and durable project knowledge belong in repository documents. Conversational habits, temporary preferences, and ephemeral working context belong here.

### Why storage lives under user app-data but remains project-scoped

The runtime behaves like a normal user-level application, so its default SQLite state belongs under user app-data rather than inside each consuming repository. That default also fits the kind of data Breathing Memory stores: conversational history, personal working habits, temporary preferences, and other user-level memory that should follow the user rather than live in version-controlled project files. At the same time, remembered content is isolated by project identity so unrelated repositories and directories do not share one memory space by accident.

### Why capture is explicit MCP rather than runtime-log import

Explicit MCP calls make remembered input intentional and inspectable. Runtime artifacts such as archived JSONL files are implementation details and are too unstable to serve as the primary operating path.

## Representation Choices

### Why `anchor` and `fragment` are separate

An `anchor` represents the semantic conversation item being answered. A `fragment` is one concrete realization of that item. The split is needed because compression can create additional child fragments while reply structure and source-side continuity still belong to one semantic node.

### Why `fragments` are live-only

Forgotten fragments are physically purged instead of being kept as soft-deleted rows. This keeps the concrete fragment table simple and current. Any maintenance history that still matters after purge is summarized into aggregates such as `sequence_metrics`.

### Why edits are modeled as forks

Editing an earlier agent message is treated as creating a new branch, not overwriting the old one. Overwrite semantics would rewrite past references, feedback, and compression outcomes. A fork preserves append-order history and keeps prior branches inspectable.

## Compression Architecture

### Why compression quality does not use a semantic parent-child gate

Compression already participates in the forgetting model through the existing acceptance and survival rules: a child must be meaningfully shorter, failed attempts increase the parent's failure penalty, and surviving children still have to prove their value later through retrieval and maintenance pressure. Adding a separate semantic parent-child gate would introduce another threshold with its own failure behavior and would make compression policy less consistent with the rest of the model.

### Why compression does not pin a separate Codex model by default

Compression runs inside the same user-visible collaboration flow as the rest of agent work. Hard-coding a separate model or reasoning profile for compression would make that path drift away from the user's active Codex defaults, increase maintenance risk when client options evolve, and create a second place where style or tone can diverge unexpectedly. If compression-specific overrides are ever needed, they should be explicit opt-in configuration rather than hidden hard-coded defaults.

## Retrieval And Attribution

### Why references only count material use

Search hits alone are predictions. Breathing Memory records a reference only when the final answer actually uses a remembered fragment. This keeps reference growth tied to concrete use instead of to broad retrieval noise.

### Why feedback attribution is delegated to the agent

User replies often evaluate either the immediately preceding answer, one of the remembered fragments used in that answer, or both. A rigid fixed rule would misattribute many cases. Letting the agent attribute feedback keeps the system aligned with future model improvements.

### Why observation wins over retrieval prediction

Retrieval ranking is only a prediction about what should matter. A concrete final-answer reference is an observation that a fragment did matter. When the two disagree, the observed use is treated as the stronger signal.

## Retrieval Architecture

### Why semantic retrieval uses a pluggable embedding backend

Embedding generation is intentionally separated from storage, indexing, and reranking. This keeps the backend boundary narrow and prevents the public retrieval API from depending on storage or indexing details. The current implementation ships with a single default sentence-transformers model; model replacement remains an internal implementation concern rather than a public multi-provider feature.

### Why the internal architecture separates core service logic from transport adapters

Breathing Memory now treats the MCP server as one adapter, not as the product boundary itself. The memory model, retrieval flow, and maintenance logic need to remain reusable from in-process consumers without forcing those consumers through MCP-shaped call paths. Keeping MCP as a thin adapter preserves the stable external tool surface for Codex while making the core service directly reusable from other runtimes.

### Why payload serialization lives in adapters instead of core types

Typed core results are meant to represent the service contract, not one transport's wire format. Keeping JSON-like payload assembly in adapters avoids leaking MCP compatibility concerns back into the core layer and lets in-process consumers depend on typed objects directly.

### Why in-process composition has a formal factory

Non-MCP consumers still need a stable way to get the default project-scoped config and instantiate the typed core engine. Putting that bootstrap path behind a small public factory keeps CLI wiring and in-process wiring aligned without pushing transport-specific concerns back into the core service contract.

### Why `install-codex` prefers smart defaults over many explicit modes

`install-codex` is an onboarding command, not a day-to-day power tool. Its default behavior therefore prioritizes low-friction success over exhaustive upfront choice. Breathing Memory keeps the user-level Codex config as the default target, because that remains the simplest mental model for most installs. At the same time, the implementation is allowed to stay flexible internally so it can detect existing registrations, avoid unnecessary rewrites, and support repository-local Codex config when the caller explicitly asks for it. Detailed state explanation belongs in `doctor`, not in a growing list of installer flags.

### Why user-facing setup is framed as `super_lite` vs `default` while `lite` remains internal

For user-facing setup, the main meaningful boundary is whether the semantic stack is installed at all. That is the real onboarding cliff. A no-extra install maps naturally to `super_lite`, and a semantic install maps naturally to `default` when the full stack is available. Runtime still needs an intermediate fallback when embeddings exist but HNSW support is unavailable, so `lite` remains a useful internal state and a useful diagnostic label. It is not, however, the primary user-facing setup target, because exposing it as a first-class onboarding path would add conceptual weight without reducing the real installation barrier.
