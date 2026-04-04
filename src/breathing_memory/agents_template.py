from __future__ import annotations

import importlib.util

from .config import MemoryConfig


AGENTS_BLOCK_START = "<!-- BEGIN BREATHING MEMORY -->"
AGENTS_BLOCK_END = "<!-- END BREATHING MEMORY -->"
AGENTS_BLOCK_PREFIX = """## Breathing Memory

This repository uses Breathing Memory during agent work in this repository.

### Required Flow

For every user turn in this repository, the agent must use the MCP tools in this order:

1. If the immediately previous final agent answer has not yet been remembered, check `memory_recent` for the latest remembered `agent` fragment with the same `reply_to`; if the latest remembered fragment already has the same content, skip the duplicate save, otherwise save it with `memory_remember(actor="agent")`
2. Check `memory_recent` before saving the current user message; use `reply_to + content` as the first duplicate check, and if needed use a very recent `actor + content` fallback before calling `memory_remember(actor="user")`
3. Immediately after saving the current user message, call `memory_read_active_collaboration_policy()` before any other tool call
4. If needed for contextual understanding, continuity, or answer accuracy, use `memory_search` after ACP and before other substantive exploration

Use the returned previous-agent `anchor_id` as the current user's `reply_to` when the user is replying to the immediately previous answer.
When the user is replying or forking from an earlier remembered anchor, pass that target as the user's `reply_to` instead.
For a root user message, omit `reply_to`.

### What To Save

- Save every user message.
- Save each final user-facing answer on the next user turn.
- Do not save intermediary commentary, progress updates, or tool-status messages.
- Use `memory_recent` as the caller-side first check against immediately repeated saves before calling `memory_remember`.
- Do not save duplicate retries of the same final answer.
- `memory_remember` suppresses duplicate deferred `agent` capture for the same `reply_to` and content, but callers must still pass accurate `reply_to` values and capture timing.
- For `user` messages, use caller-side `memory_recent` checks before `memory_remember` instead of relying on engine-side duplicate suppression.
- If no later user turn arrives, the final agent answer may remain unremembered.
- When a reusable rule about how to collaborate with the user becomes clear, the agent may save one or more derived `agent` fragments with `kind="collaboration_policy"`.
- This may be derived either from explicit user feedback or from broader conversational context when the caller judges it likely to be reusable.
- Prefer saving collaboration-policy fragments only when they are likely to affect future behavior, choices, or response style.
- Do not save weak inferences, one-off requests, transient emotions, or ambiguous signals as collaboration policy.
- When uncertain, prefer not to save.
- Keep each `collaboration_policy` fragment focused on a single reusable rule.
- Use the same `reply_to` as the current turn's user anchor, but treat these fragments as derived policy memory rather than conversational threading.

### Search Query
"""

AGENTS_SEARCH_QUERY_COMMON = """- Use Breathing Memory retrieval to review relevant prior interactions when that would improve contextual understanding, continuity, or answer accuracy for the current user request.
- `memory_search.query` must be chosen by the MCP-calling agent for the current user request.
- Keep the query in the user's language and avoid unnecessary translation.
- Use the default `search_effort` of `32` unless there is a concrete reason to choose a different valid value up front.
- Start with a `result_count` of `4` unless there is a concrete reason to choose a different valid value up front.
- If retrieval is cleaner when limited to one side of the conversation, `memory_search` may use `actor="user"` or `actor="agent"`.
- If the first search result looks insufficient, rerun `memory_search` as many times as needed with a broader `result_count`, a higher `search_effort`, or both.
- Treat `result_count` as powers of two from the base `4`, and `search_effort` as powers of two from the base `32`.
"""

AGENTS_SEARCH_QUERY_SUPER_LITE = """- Choose a query optimized for lexical retrieval.
- Rewrite the user request into a shorter search-oriented query when that improves lexical matching.
- Use keyword- or phrase-oriented queries when they improve lexical retrieval.
"""

AGENTS_SEARCH_QUERY_SEMANTIC = """- Choose a query optimized for semantic retrieval.
- Rewrite the user request into a search-oriented query when that improves semantic retrieval.
- Do not collapse the query into a keyword bag unless there is a clear retrieval benefit.
"""

AGENTS_BLOCK_SUFFIX = """

### Source References

- Track which fragments returned by `memory_search` or `memory_read_active_collaboration_policy` materially inform the final answer while drafting it.
- If the deferred final answer materially uses fragments returned by `memory_search` or `memory_read_active_collaboration_policy`, pass those fragment ids as `source_fragment_ids` when that answer is persisted on the next user turn.
- If no retrieved fragment materially informed the final answer, omit `source_fragment_ids`.

### Response Footer

- In every final user-facing answer, append a single-line footer after the main response body.
- Insert a horizontal rule line immediately before the footer.
- Use this exact shape:
  `---`
  `BM: ok | agents=checked | user_anchor=... | acp=... | search=... | refs=...`
- `agents=checked` is a self-check that the agent reread or actively followed this AGENTS guidance for the current turn.
- `user_anchor` is the current user message anchor id for this turn.
- `acp` is the count returned by `memory_read_active_collaboration_policy`.
- `search` is the count returned by `memory_search`.
- `refs` is the number of fragment ids actually passed as `source_fragment_ids` for the deferred final answer; use `0` when none were materially used.
- Do not add debug-only fields such as `prev_agent_anchor`, `feedbacks`, or per-source breakdowns in the standard footer.

### Collaboration Policy

- Use `memory_read_active_collaboration_policy()` to preload collaboration-policy memory before forming the task query and answering.
- Use collaboration-policy memory to shape how to answer or proceed, not as a substitute for task-memory retrieval.
- If remembered collaboration context seems relevant but uncertain, the agent may confirm it with the user before relying on it.
- If needed, run an additional `memory_search(..., kind="collaboration_policy")` for targeted clarification.

### Feedback Attribution

- When a user message clearly confirms, corrects, or evaluates remembered information, record that with `memory_feedback`.
- Decide whether that feedback applies to the immediately previous answer fragment, to referenced fragments used by that answer, or to both.
- If the target of the feedback is ambiguous, skip `memory_feedback` rather than guessing.

### Failure Policy

- Do not fabricate remembered ids such as `reply_to` or `source_fragment_ids`.
- If semantic-index mode is enabled and the semantic index is being rebuilt or recovered, do not issue other Breathing Memory mutations or semantic searches until that rebuild completes.
- `archived_sessions/*.jsonl` and other runtime files are not the primary capture path. They are internal implementation details and must not be used as the default memory source.
"""


def render_agents_block(*, guidance_mode: str) -> str:
    return f"{AGENTS_BLOCK_START}\n{build_agents_block_body(guidance_mode).rstrip()}\n{AGENTS_BLOCK_END}"


def build_agents_block_body(guidance_mode: str) -> str:
    if guidance_mode == "super_lite":
        mode_specific = AGENTS_SEARCH_QUERY_SUPER_LITE
    elif guidance_mode == "semantic":
        mode_specific = AGENTS_SEARCH_QUERY_SEMANTIC
    else:
        raise ValueError("guidance_mode must be 'super_lite' or 'semantic'")
    return (
        AGENTS_BLOCK_PREFIX
        + AGENTS_SEARCH_QUERY_COMMON
        + mode_specific
        + AGENTS_BLOCK_SUFFIX
    )


def resolve_agents_guidance_mode(
    retrieval_mode: str | None = None,
    semantic_available: bool | None = None,
) -> str:
    active_retrieval_mode = MemoryConfig().retrieval_mode if retrieval_mode is None else retrieval_mode
    has_semantic_support = semantic_extra_available() if semantic_available is None else semantic_available

    if active_retrieval_mode == "super_lite":
        return "super_lite"
    if active_retrieval_mode in {"lite", "default"}:
        return "semantic"
    if active_retrieval_mode == "auto":
        return "semantic" if has_semantic_support else "super_lite"
    raise ValueError("retrieval_mode must be 'auto', 'super_lite', 'lite', or 'default'")


def semantic_extra_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None
