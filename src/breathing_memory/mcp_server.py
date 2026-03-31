from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any, AsyncIterator, Callable

import mcp.server.stdio
from mcp import types
from mcp.server import Server

from .config import MemoryConfig
from .engine import BreathingMemoryEngine


SERVER_NAME = "breathing-memory"
SERVER_INSTRUCTIONS = (
    "Breathing Memory provides tools for persisting and retrieving collaboration memory. "
    "Use memory_remember to persist a turn, memory_search to retrieve relevant fragments, "
    "memory_read_active_collaboration_policy to load collaboration-policy fragments before answering, "
    "memory_fetch for direct lookup, memory_recent to inspect the latest remembered root fragments, "
    "memory_feedback to record evaluation, and memory_stats for diagnostics."
)

ToolHandler = Callable[[BreathingMemoryEngine, dict[str, Any]], dict[str, Any]]


def _package_version() -> str:
    try:
        return version("breathing-memory")
    except PackageNotFoundError:
        return "0.5.5"


def _tool_definitions() -> list[types.Tool]:
    return [
        types.Tool(
            name="memory_remember",
            description="Persist one remembered fragment and any material references used in the final answer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "actor": {"type": "string", "enum": ["user", "agent"]},
                    "reply_to": {"type": ["integer", "null"]},
                    "source_fragment_ids": {"type": "array", "items": {"type": "integer"}},
                    "kind": {"type": ["string", "null"]},
                },
                "required": ["content", "actor"],
            },
        ),
        types.Tool(
            name="memory_search",
            description="Retrieve remembered fragments by lexical match and rerank by search priority.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "result_count": {"type": "integer", "minimum": 8},
                    "search_effort": {"type": "integer", "minimum": 32},
                    "actor": {"type": "string", "enum": ["user", "agent"]},
                    "kind": {"type": ["string", "null"]},
                    "include_diagnostics": {"type": "boolean"},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="memory_read_active_collaboration_policy",
            description="Load active collaboration-policy fragments within a token budget.",
            inputSchema={
                "type": "object",
                "properties": {
                    "token_budget": {"type": "integer", "minimum": 1},
                },
            },
        ),
        types.Tool(
            name="memory_fetch",
            description="Directly fetch remembered fragments by fragment id or anchor id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fragment_id": {"type": "integer"},
                    "anchor_id": {"type": "integer"},
                },
            },
        ),
        types.Tool(
            name="memory_recent",
            description="Fetch the most recent remembered root fragments, optionally filtered by actor and reply target.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1},
                    "actor": {"type": "string", "enum": ["user", "agent"]},
                    "reply_to": {"type": "integer"},
                },
            },
        ),
        types.Tool(
            name="memory_feedback",
            description="Record explicit or attributable evaluation of a concrete remembered fragment.",
            inputSchema={
                "type": "object",
                "properties": {
                    "from_anchor_id": {"type": "integer"},
                    "fragment_id": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                },
                "required": ["from_anchor_id", "fragment_id", "verdict"],
            },
        ),
        types.Tool(
            name="memory_stats",
            description="Inspect current memory state and effective parameter values.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


def _remember_handler(engine: BreathingMemoryEngine, arguments: dict[str, Any]) -> dict[str, Any]:
    return engine.remember(
        content=str(arguments["content"]),
        actor=str(arguments["actor"]),
        reply_to=arguments.get("reply_to"),
        source_fragment_ids=arguments.get("source_fragment_ids"),
        kind=arguments.get("kind"),
    )


def _search_handler(engine: BreathingMemoryEngine, arguments: dict[str, Any]) -> dict[str, Any]:
    return engine.search(
        query=str(arguments["query"]),
        result_count=arguments.get("result_count"),
        search_effort=arguments.get("search_effort"),
        actor=arguments.get("actor"),
        kind=arguments.get("kind"),
        include_diagnostics=bool(arguments.get("include_diagnostics", False)),
    )


def _read_active_collaboration_policy_handler(
    engine: BreathingMemoryEngine,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return engine.read_active_collaboration_policy(
        token_budget=arguments.get("token_budget"),
    )


def _fetch_handler(engine: BreathingMemoryEngine, arguments: dict[str, Any]) -> dict[str, Any]:
    return engine.fetch(
        fragment_id=arguments.get("fragment_id"),
        anchor_id=arguments.get("anchor_id"),
    )


def _feedback_handler(engine: BreathingMemoryEngine, arguments: dict[str, Any]) -> dict[str, Any]:
    return engine.feedback(
        from_anchor_id=int(arguments["from_anchor_id"]),
        fragment_id=int(arguments["fragment_id"]),
        verdict=str(arguments["verdict"]),
    )


def _recent_handler(engine: BreathingMemoryEngine, arguments: dict[str, Any]) -> dict[str, Any]:
    return engine.recent(
        limit=int(arguments.get("limit", 4)),
        actor=arguments.get("actor"),
        reply_to=arguments.get("reply_to"),
    )


def _stats_handler(engine: BreathingMemoryEngine, arguments: dict[str, Any]) -> dict[str, Any]:
    del arguments
    return engine.stats()


TOOL_HANDLERS: dict[str, ToolHandler] = {
    "memory_remember": _remember_handler,
    "memory_search": _search_handler,
    "memory_read_active_collaboration_policy": _read_active_collaboration_policy_handler,
    "memory_fetch": _fetch_handler,
    "memory_recent": _recent_handler,
    "memory_feedback": _feedback_handler,
    "memory_stats": _stats_handler,
}


@asynccontextmanager
async def _managed_engine(
    config: MemoryConfig | None = None,
    engine: BreathingMemoryEngine | None = None,
) -> AsyncIterator[BreathingMemoryEngine]:
    if engine is not None:
        yield engine
        return

    managed_engine = BreathingMemoryEngine(config=config or MemoryConfig())
    try:
        yield managed_engine
    finally:
        managed_engine.close()


def create_mcp_server(
    *,
    config: MemoryConfig | None = None,
    engine: BreathingMemoryEngine | None = None,
) -> Server[BreathingMemoryEngine, Any]:
    @asynccontextmanager
    async def lifespan(_server: Server[BreathingMemoryEngine, Any]) -> AsyncIterator[BreathingMemoryEngine]:
        async with _managed_engine(config=config, engine=engine) as active_engine:
            yield active_engine

    server: Server[BreathingMemoryEngine, Any] = Server(
        SERVER_NAME,
        version=_package_version(),
        instructions=SERVER_INSTRUCTIONS,
        lifespan=lifespan,
    )

    @server.list_tools()
    async def handle_list_tools(request: types.ListToolsRequest) -> types.ListToolsResult:
        del request
        return types.ListToolsResult(tools=_tool_definitions())

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")
        active_engine = server.request_context.lifespan_context
        return handler(active_engine, arguments)

    return server


async def serve_stdio_server(
    *,
    config: MemoryConfig | None = None,
    engine: BreathingMemoryEngine | None = None,
) -> None:
    server = create_mcp_server(config=config, engine=engine)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def serve_stdio(
    *,
    config: MemoryConfig | None = None,
    engine: BreathingMemoryEngine | None = None,
) -> None:
    asyncio.run(serve_stdio_server(config=config, engine=engine))
