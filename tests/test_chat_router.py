"""
Integration tests for routers/chat.py  (POST /v1/chat/completions)

The upstream API is mocked by patching `routers.chat.stream_upstream_response`
(the imported name, not the original module path).  This lets us test the
entire request→conversion→SSE-formatting pipeline without a real upstream server.
"""

from __future__ import annotations

import json
from typing import AsyncIterator
from unittest.mock import patch

from httpx import AsyncClient

from tests.conftest import iter_sse_chunks


# ── Mock stream factories ─────────────────────────────────────────────────────


def make_text_stream(*chunks: str):
    """Return an async-generator function that yields the given text chunks."""

    async def _stream(*_args, **_kwargs) -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    return _stream


def make_tool_call_stream(tool_name: str, arguments: str, call_id: str = "call_test1"):
    """Return an async-generator that yields a single JSON tool-call response."""
    payload = json.dumps(
        {
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": arguments},
                }
            ]
        }
    )
    # Yield in multiple chunks to exercise the JSON buffering path
    mid = len(payload) // 2
    return make_text_stream(payload[:mid], payload[mid:])


# ── Shared request body ───────────────────────────────────────────────────────

SIMPLE_BODY = {
    "model": "gateway-adapter",
    "messages": [{"role": "user", "content": "Hello"}],
}

TOOL_BODY = {
    "model": "gateway-adapter",
    "messages": [{"role": "user", "content": "What's 2+2?"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluates a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }
    ],
}


# ── Non-streaming tests ───────────────────────────────────────────────────────


class TestNonStreaming:
    async def test_simple_text_response(self, client: AsyncClient):
        with patch(
            "routers.chat.stream_upstream_response", new=make_text_stream("Hi there!")
        ):
            resp = await client.post("/v1/chat/completions", json=SIMPLE_BODY)

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hi there!"
        assert data["choices"][0]["finish_reason"] == "stop"

    async def test_multi_chunk_text_concatenated(self, client: AsyncClient):
        with patch(
            "routers.chat.stream_upstream_response",
            new=make_text_stream("Hello", ", ", "world"),
        ):
            resp = await client.post("/v1/chat/completions", json=SIMPLE_BODY)

        assert resp.json()["choices"][0]["message"]["content"] == "Hello, world"

    async def test_tool_call_response_non_streaming(self, client: AsyncClient):
        with patch(
            "routers.chat.stream_upstream_response",
            new=make_tool_call_stream("calculate", '{"expression": "2+2"}'),
        ):
            resp = await client.post("/v1/chat/completions", json=TOOL_BODY)

        assert resp.status_code == 200
        data = resp.json()
        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        tc = choice["message"]["tool_calls"][0]
        assert tc["function"]["name"] == "calculate"
        assert tc["id"] == "call_test1"

    async def test_response_has_required_openai_fields(self, client: AsyncClient):
        with patch("routers.chat.stream_upstream_response", new=make_text_stream("ok")):
            resp = await client.post("/v1/chat/completions", json=SIMPLE_BODY)

        data = resp.json()
        for field in ("id", "object", "created", "model", "choices", "usage"):
            assert field in data, f"Missing field: {field}"

    async def test_model_name_echoed_back(self, client: AsyncClient):
        with patch("routers.chat.stream_upstream_response", new=make_text_stream("ok")):
            resp = await client.post("/v1/chat/completions", json=SIMPLE_BODY)
        assert resp.json()["model"] == "gateway-adapter"

    async def test_usage_zeros_returned(self, client: AsyncClient):
        with patch("routers.chat.stream_upstream_response", new=make_text_stream("ok")):
            resp = await client.post("/v1/chat/completions", json=SIMPLE_BODY)
        usage = resp.json()["usage"]
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0


# ── Streaming tests ───────────────────────────────────────────────────────────


class TestStreaming:
    STREAM_BODY = {**SIMPLE_BODY, "stream": True}

    async def test_returns_event_stream_content_type(self, client: AsyncClient):
        with patch("routers.chat.stream_upstream_response", new=make_text_stream("Hi")):
            async with client.stream(
                "POST", "/v1/chat/completions", json=self.STREAM_BODY
            ) as resp:
                assert resp.status_code == 200
                ct = resp.headers["content-type"]
                assert "text/event-stream" in ct

    async def test_first_chunk_contains_role_delta(self, client: AsyncClient):
        with patch("routers.chat.stream_upstream_response", new=make_text_stream("Hi")):
            async with client.stream(
                "POST", "/v1/chat/completions", json=self.STREAM_BODY
            ) as resp:
                chunks = [c async for c in iter_sse_chunks(resp)]

        first = chunks[0]
        assert first["choices"][0]["delta"].get("role") == "assistant"

    async def test_content_chunks_streamed(self, client: AsyncClient):
        with patch(
            "routers.chat.stream_upstream_response",
            new=make_text_stream("Hello", " world"),
        ):
            async with client.stream(
                "POST", "/v1/chat/completions", json=self.STREAM_BODY
            ) as resp:
                chunks = [c async for c in iter_sse_chunks(resp)]

        content_chunks = [c for c in chunks if c["choices"][0]["delta"].get("content")]
        full_text = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)
        assert full_text == "Hello world"

    async def test_final_chunk_has_finish_reason_stop(self, client: AsyncClient):
        with patch("routers.chat.stream_upstream_response", new=make_text_stream("ok")):
            async with client.stream(
                "POST", "/v1/chat/completions", json=self.STREAM_BODY
            ) as resp:
                chunks = [c async for c in iter_sse_chunks(resp)]

        finish_chunks = [
            c for c in chunks if c["choices"][0].get("finish_reason") == "stop"
        ]
        assert len(finish_chunks) == 1

    async def test_all_chunks_have_same_id(self, client: AsyncClient):
        with patch(
            "routers.chat.stream_upstream_response", new=make_text_stream("a", "b")
        ):
            async with client.stream(
                "POST", "/v1/chat/completions", json=self.STREAM_BODY
            ) as resp:
                chunks = [c async for c in iter_sse_chunks(resp)]

        ids = {c["id"] for c in chunks}
        assert len(ids) == 1  # same completion ID for all chunks

    async def test_all_chunks_have_chat_completion_chunk_object(
        self, client: AsyncClient
    ):
        with patch("routers.chat.stream_upstream_response", new=make_text_stream("hi")):
            async with client.stream(
                "POST", "/v1/chat/completions", json=self.STREAM_BODY
            ) as resp:
                chunks = [c async for c in iter_sse_chunks(resp)]

        for chunk in chunks:
            assert chunk["object"] == "chat.completion.chunk"

    async def test_streaming_tool_call_phase1_has_name_and_id(
        self, client: AsyncClient
    ):
        stream_body = {**TOOL_BODY, "stream": True}
        with patch(
            "routers.chat.stream_upstream_response",
            new=make_tool_call_stream("calculate", '{"expression": "2+2"}', "call_abc"),
        ):
            async with client.stream(
                "POST", "/v1/chat/completions", json=stream_body
            ) as resp:
                chunks = [c async for c in iter_sse_chunks(resp)]

        tool_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
        assert len(tool_chunks) >= 2  # at least phase1 + phase2

        # Phase 1: name present, id present, arguments empty
        phase1 = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        assert phase1["function"]["name"] == "calculate"
        assert phase1["id"] == "call_abc"
        assert phase1["function"]["arguments"] == ""

    async def test_streaming_tool_call_phase2_has_arguments(self, client: AsyncClient):
        stream_body = {**TOOL_BODY, "stream": True}
        with patch(
            "routers.chat.stream_upstream_response",
            new=make_tool_call_stream("calculate", '{"expression": "2+2"}'),
        ):
            async with client.stream(
                "POST", "/v1/chat/completions", json=stream_body
            ) as resp:
                chunks = [c async for c in iter_sse_chunks(resp)]

        tool_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
        # Phase 2: arguments filled, id absent
        phase2 = tool_chunks[1]["choices"][0]["delta"]["tool_calls"][0]
        assert phase2["function"]["arguments"] == '{"expression": "2+2"}'
        assert "id" not in phase2 or phase2.get("id") is None

    async def test_streaming_tool_call_finish_reason_is_tool_calls(
        self, client: AsyncClient
    ):
        stream_body = {**TOOL_BODY, "stream": True}
        with patch(
            "routers.chat.stream_upstream_response",
            new=make_tool_call_stream("calculate", "{}"),
        ):
            async with client.stream(
                "POST", "/v1/chat/completions", json=stream_body
            ) as resp:
                chunks = [c async for c in iter_sse_chunks(resp)]

        finish = [
            c for c in chunks if c["choices"][0].get("finish_reason") == "tool_calls"
        ]
        assert len(finish) == 1

    async def test_json_that_is_not_tool_calls_streamed_as_content(
        self, client: AsyncClient
    ):
        """Edge case: model emits JSON that isn't a tool_calls payload."""
        stream_body = {**SIMPLE_BODY, "stream": True}
        with patch(
            "routers.chat.stream_upstream_response",
            new=make_text_stream('{"answer": 42}'),
        ):
            async with client.stream(
                "POST", "/v1/chat/completions", json=stream_body
            ) as resp:
                chunks = [c async for c in iter_sse_chunks(resp)]

        # Should NOT be a tool call
        assert not any(c["choices"][0]["delta"].get("tool_calls") for c in chunks)
        # Should be emitted as content
        content_chunks = [c for c in chunks if c["choices"][0]["delta"].get("content")]
        assert content_chunks
        finish = [c for c in chunks if c["choices"][0].get("finish_reason") == "stop"]
        assert finish


# ── Utility endpoint tests ────────────────────────────────────────────────────


class TestUtilityEndpoints:
    async def test_health_ok(self, client: AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_models_list_not_empty(self, client: AsyncClient):
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1

    async def test_root_lists_endpoints(self, client: AsyncClient):
        resp = await client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "chat_completions" in body.get("endpoints", {})
