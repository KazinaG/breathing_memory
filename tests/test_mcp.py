from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import anyio
from mcp import types
from mcp.client.session import ClientSession

from breathing_memory.config import MemoryConfig
from breathing_memory.engine import BreathingMemoryEngine
from breathing_memory.mcp_server import create_mcp_server


class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        config = MemoryConfig(
            db_path=Path(self.tempdir.name) / "memory.sqlite3",
            total_capacity_mb=120 / (1024 * 1024),
            retrieval_mode="super_lite",
        )
        self.engine = BreathingMemoryEngine(config=config)

    def tearDown(self) -> None:
        self.engine.close()
        self.tempdir.cleanup()

    async def _with_session(self, callback):
        server = create_mcp_server(engine=self.engine)
        client_to_server_send, client_to_server_recv = anyio.create_memory_object_stream(0)
        server_to_client_send, server_to_client_recv = anyio.create_memory_object_stream(0)

        async with anyio.create_task_group() as tg:
            tg.start_soon(
                server.run,
                client_to_server_recv,
                server_to_client_send,
                server.create_initialization_options(),
                True,
            )

            async with ClientSession(server_to_client_recv, client_to_server_send) as session:
                init = await session.initialize()
                await session.list_tools()
                result = await callback(session, init)

            tg.cancel_scope.cancel()

        return result

    async def test_tools_list_exposes_seven_tools(self) -> None:
        async def callback(session: ClientSession, init: types.InitializeResult):
            tools = await session.list_tools()
            return init, tools

        init, tools = await self._with_session(callback)
        self.assertEqual(init.protocolVersion, types.LATEST_PROTOCOL_VERSION)
        self.assertEqual(len(tools.tools), 7)
        names = {tool.name for tool in tools.tools}
        self.assertEqual(
            names,
            {
                "memory_remember",
                "memory_search",
                "memory_read_active_collaboration_policy",
                "memory_fetch",
                "memory_recent",
                "memory_feedback",
                "memory_stats",
            },
        )

    async def test_list_tools_starts_background_embedding_warmup(self) -> None:
        calls: list[str] = []
        original = self.engine.start_background_embedding_warmup

        def _wrapped() -> bool:
            calls.append("started")
            return original()

        self.engine.start_background_embedding_warmup = _wrapped  # type: ignore[method-assign]
        try:
            async def callback(session: ClientSession, init: types.InitializeResult):
                del init
                return await session.list_tools()

            tools = await self._with_session(callback)
        finally:
            self.engine.start_background_embedding_warmup = original  # type: ignore[method-assign]

        self.assertEqual(len(tools.tools), 7)
        self.assertEqual(calls, ["started", "started"])

    async def test_memory_tool_flow(self) -> None:
        async def callback(session: ClientSession, init: types.InitializeResult):
            del init
            remember = await session.call_tool(
                "memory_remember",
                {
                    "content": "hello memory",
                    "actor": "user",
                },
            )
            remembered = remember.structuredContent
            assert isinstance(remembered, dict)

            search = await session.call_tool(
                "memory_search",
                {"query": "hello", "result_count": 4, "search_effort": 32},
            )

            fetch = await session.call_tool(
                "memory_fetch",
                {"fragment_id": remembered["id"]},
            )

            feedback = await session.call_tool(
                "memory_feedback",
                {
                    "from_anchor_id": remembered["anchor_id"],
                    "fragment_id": remembered["id"],
                    "verdict": "negative",
                },
            )

            stats = await session.call_tool("memory_stats", {})
            return remember, search, fetch, feedback, stats

        remember, search, fetch, feedback, stats = await self._with_session(callback)
        self.assertFalse(remember.isError)
        self.assertIsInstance(remember.structuredContent, dict)
        self.assertEqual(search.structuredContent["count"], 1)
        self.assertFalse(fetch.isError)
        self.assertEqual(fetch.structuredContent["count"], 1)
        self.assertFalse(feedback.isError)
        self.assertEqual(feedback.structuredContent["confidence_score"], 0.5)
        self.assertEqual(stats.structuredContent["fragment_count"], 1)

    async def test_memory_search_can_include_diagnostics(self) -> None:
        async def callback(session: ClientSession, init: types.InitializeResult):
            del init
            await session.call_tool(
                "memory_remember",
                {
                    "content": "hello memory",
                    "actor": "user",
                },
            )
            return await session.call_tool(
                "memory_search",
                {
                    "query": "hello",
                    "result_count": 4,
                    "search_effort": 32,
                    "include_diagnostics": True,
                },
            )

        result = await self._with_session(callback)
        self.assertFalse(result.isError)
        item = result.structuredContent["items"][0]
        self.assertIn("diagnostics", item)
        self.assertEqual(item["diagnostics"]["retrieval_mode"], "super_lite")

    async def test_memory_search_can_filter_by_actor(self) -> None:
        async def callback(session: ClientSession, init: types.InitializeResult):
            del init
            await session.call_tool(
                "memory_remember",
                {
                    "content": "shared summary",
                    "actor": "user",
                },
            )
            remembered = await session.call_tool(
                "memory_remember",
                {
                    "content": "shared summary",
                    "actor": "agent",
                },
            )
            return remembered, await session.call_tool(
                "memory_search",
                {
                    "query": "shared summary",
                    "result_count": 4,
                    "search_effort": 32,
                    "actor": "agent",
                },
            )

        remembered, result = await self._with_session(callback)
        self.assertFalse(result.isError)
        self.assertEqual(result.structuredContent["count"], 1)
        self.assertEqual(result.structuredContent["items"][0]["id"], remembered.structuredContent["id"])
        self.assertEqual(result.structuredContent["items"][0]["actor"], "agent")

    async def test_memory_read_active_collaboration_policy_returns_budgeted_items(self) -> None:
        async def callback(session: ClientSession, init: types.InitializeResult):
            del init
            await session.call_tool(
                "memory_remember",
                {
                    "content": "General answer",
                    "actor": "agent",
                },
            )
            remembered = await session.call_tool(
                "memory_remember",
                {
                    "content": "Detailed collaboration guidance for future turns.",
                    "actor": "agent",
                    "kind": "collaboration_policy",
                },
            )
            result = await session.call_tool(
                "memory_read_active_collaboration_policy",
                {"token_budget": 1},
            )
            return remembered, result

        remembered, result = await self._with_session(callback)
        self.assertFalse(result.isError)
        self.assertEqual(result.structuredContent["count"], 1)
        item = result.structuredContent["items"][0]
        self.assertEqual(item["id"], remembered.structuredContent["id"])
        self.assertEqual(item["kind"], "collaboration_policy")
        self.assertGreater(result.structuredContent["used_token_budget"], 1)

    async def test_memory_remember_rejects_unknown_reply_to(self) -> None:
        async def callback(session: ClientSession, init: types.InitializeResult):
            del init
            return await session.call_tool(
                "memory_remember",
                {
                    "content": "hello memory",
                    "actor": "user",
                    "reply_to": 9999,
                },
            )

        result = await self._with_session(callback)
        self.assertTrue(result.isError)
        self.assertIn("reply_to anchor not found", result.content[0].text)

    async def test_memory_fetch_requires_exactly_one_selector(self) -> None:
        async def callback(session: ClientSession, init: types.InitializeResult):
            del init
            return await session.call_tool("memory_fetch", {})

        result = await self._with_session(callback)
        self.assertTrue(result.isError)
        self.assertIn("exactly one of fragment_id or anchor_id", result.content[0].text)

    async def test_memory_remember_deduplicates_agent_capture(self) -> None:
        async def callback(session: ClientSession, init: types.InitializeResult):
            del init
            parent = await session.call_tool(
                "memory_remember",
                {
                    "content": "question",
                    "actor": "user",
                },
            )
            parent_content = parent.structuredContent
            assert isinstance(parent_content, dict)

            first = await session.call_tool(
                "memory_remember",
                {
                    "content": "same answer",
                    "actor": "agent",
                    "reply_to": parent_content["anchor_id"],
                },
            )
            second = await session.call_tool(
                "memory_remember",
                {
                    "content": "same answer",
                    "actor": "agent",
                    "reply_to": parent_content["anchor_id"],
                },
            )
            stats = await session.call_tool("memory_stats", {})
            return first, second, stats

        first, second, stats = await self._with_session(callback)
        self.assertFalse(first.isError)
        self.assertFalse(second.isError)
        self.assertEqual(first.structuredContent["id"], second.structuredContent["id"])
        self.assertEqual(stats.structuredContent["fragment_count"], 2)

    async def test_memory_remember_keeps_distinct_user_capture(self) -> None:
        async def callback(session: ClientSession, init: types.InitializeResult):
            del init
            parent = await session.call_tool(
                "memory_remember",
                {
                    "content": "question",
                    "actor": "agent",
                },
            )
            parent_content = parent.structuredContent
            assert isinstance(parent_content, dict)

            first = await session.call_tool(
                "memory_remember",
                {
                    "content": "same follow-up",
                    "actor": "user",
                    "reply_to": parent_content["anchor_id"],
                },
            )
            second = await session.call_tool(
                "memory_remember",
                {
                    "content": "same follow-up",
                    "actor": "user",
                    "reply_to": parent_content["anchor_id"],
                },
            )
            stats = await session.call_tool("memory_stats", {})
            return first, second, stats

        first, second, stats = await self._with_session(callback)
        self.assertFalse(first.isError)
        self.assertFalse(second.isError)
        self.assertNotEqual(first.structuredContent["id"], second.structuredContent["id"])
        self.assertEqual(stats.structuredContent["fragment_count"], 3)

    async def test_memory_recent_returns_latest_matching_fragments(self) -> None:
        async def callback(session: ClientSession, init: types.InitializeResult):
            del init
            root = await session.call_tool(
                "memory_remember",
                {
                    "content": "root",
                    "actor": "user",
                },
            )
            root_content = root.structuredContent
            assert isinstance(root_content, dict)
            await session.call_tool(
                "memory_remember",
                {
                    "content": "older",
                    "actor": "user",
                    "reply_to": root_content["anchor_id"],
                },
            )
            newer = await session.call_tool(
                "memory_remember",
                {
                    "content": "newer",
                    "actor": "user",
                    "reply_to": root_content["anchor_id"],
                },
            )
            recent = await session.call_tool(
                "memory_recent",
                {
                    "limit": 2,
                    "actor": "user",
                    "reply_to": root_content["anchor_id"],
                },
            )
            return newer, recent

        newer, recent = await self._with_session(callback)
        self.assertFalse(recent.isError)
        self.assertEqual(recent.structuredContent["count"], 2)
        self.assertEqual(recent.structuredContent["items"][0]["id"], newer.structuredContent["id"])


if __name__ == "__main__":
    unittest.main()
