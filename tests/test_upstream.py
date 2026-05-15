"""
Unit tests for services/upstream.py

Tests primarily focus on _extract_text_from_sse_data — the multi-format SSE
parser — plus narrow HTTP-client edge cases around streaming response handling.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

import services.upstream as upstream
from services.upstream import _extract_text_from_sse_data, _extract_usage_from_sse_data


class _AsyncSingleChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunk: bytes) -> None:
        self._chunk = chunk

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._chunk


class TestExtractTextFromSseData:
    # ── Claude native format ─────────────────────────────────────────────────

    def test_claude_content_block_delta(self):
        data = json.dumps(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            }
        )
        assert _extract_text_from_sse_data(data) == "Hello"

    def test_claude_non_text_delta_returns_none(self):
        """input_json_delta events should not leak raw bytes as text."""
        data = json.dumps(
            {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": '{"x"'},
            }
        )
        assert _extract_text_from_sse_data(data) is None

    def test_claude_message_start_returns_none(self):
        data = json.dumps({"type": "message_start", "message": {"id": "msg_1"}})
        assert _extract_text_from_sse_data(data) is None

    def test_claude_message_stop_returns_none(self):
        data = json.dumps({"type": "message_stop"})
        assert _extract_text_from_sse_data(data) is None

    # ── Simple delta format ──────────────────────────────────────────────────

    def test_simple_delta_text(self):
        data = json.dumps({"delta": {"text": "world"}})
        assert _extract_text_from_sse_data(data) == "world"

    def test_simple_delta_empty_text_returns_none(self):
        data = json.dumps({"delta": {"text": ""}})
        assert _extract_text_from_sse_data(data) is None

    # ── Flat text format ─────────────────────────────────────────────────────

    def test_flat_text_field(self):
        data = json.dumps({"text": "flat chunk"})
        assert _extract_text_from_sse_data(data) == "flat chunk"

    def test_flat_text_empty_returns_none(self):
        data = json.dumps({"text": ""})
        assert _extract_text_from_sse_data(data) is None

    # ── OpenAI-like passthrough ───────────────────────────────────────────────

    def test_openai_like_choices_delta_content(self):
        data = json.dumps({"choices": [{"delta": {"content": "openai chunk"}}]})
        assert _extract_text_from_sse_data(data) == "openai chunk"

    def test_openai_like_empty_content_returns_none(self):
        data = json.dumps({"choices": [{"delta": {"content": ""}}]})
        assert _extract_text_from_sse_data(data) is None

    def test_openai_like_no_choices_returns_none(self):
        data = json.dumps({"choices": []})
        assert _extract_text_from_sse_data(data) is None

    # ── Raw string (no JSON wrapper) ─────────────────────────────────────────

    def test_raw_string_returned_as_is(self):
        assert _extract_text_from_sse_data("raw text chunk") == "raw text chunk"

    def test_raw_whitespace_only_returns_none(self):
        assert _extract_text_from_sse_data("   ") is None

    # ── Sentinel / empty values ───────────────────────────────────────────────

    def test_done_sentinel_returns_none(self):
        assert _extract_text_from_sse_data("[DONE]") is None

    def test_empty_string_returns_none(self):
        assert _extract_text_from_sse_data("") is None

    def test_none_input_returns_none(self):
        assert _extract_text_from_sse_data("") is None


def _make_client_factory(handler: Any) -> Any:
    """Return a drop-in replacement for ``httpx.AsyncClient`` backed by *handler*."""
    real_async_client = httpx.AsyncClient

    def make_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    return make_client


class TestStreamUpstreamResponse:
    async def test_upstream_payload_contains_only_query_and_history_strings(
        self, monkeypatch
    ):
        captured_payload: dict[str, Any] = {}
        sse_body = b'data: {"text": "ok"}\n\n'

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_payload.update(json.loads(request.content.decode()))
            return httpx.Response(
                200, stream=_AsyncSingleChunkStream(sse_body), request=request
            )

        monkeypatch.setattr(
            upstream.httpx, "AsyncClient", _make_client_factory(handler)
        )

        texts = [t async for t, _ in upstream.stream_upstream_response("Hello", [])]

        assert texts == ["ok"]
        assert set(captured_payload) == {"query", "history"}
        assert captured_payload["query"] == "Hello"
        assert captured_payload["history"] == "[]"
        assert isinstance(captured_payload["query"], str)
        assert isinstance(captured_payload["history"], str)

    async def test_upstream_history_is_json_encoded_string(self, monkeypatch):
        captured_payload: dict[str, Any] = {}
        history = [{"role": "user", "content": "이전 질문"}]
        sse_body = b'data: {"text": "ok"}\n\n'

        async def handler(request: httpx.Request) -> httpx.Response:
            captured_payload.update(json.loads(request.content.decode()))
            return httpx.Response(
                200, stream=_AsyncSingleChunkStream(sse_body), request=request
            )

        monkeypatch.setattr(
            upstream.httpx, "AsyncClient", _make_client_factory(handler)
        )

        texts = [
            t async for t, _ in upstream.stream_upstream_response("다음 질문", history)
        ]

        assert texts == ["ok"]
        assert captured_payload["query"] == "다음 질문"
        assert captured_payload["history"] == json.dumps(history, ensure_ascii=False)

    async def test_http_error_response_body_is_read_before_raise(self, monkeypatch):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                stream=_AsyncSingleChunkStream(b"quota exceeded"),
                request=request,
            )

        monkeypatch.setattr(
            upstream.httpx, "AsyncClient", _make_client_factory(handler)
        )

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            async for _text, _usage in upstream.stream_upstream_response("Hello", []):
                pass

        assert exc_info.value.response.status_code == 429
        assert exc_info.value.response.text == "quota exceeded"

    # ── Retry behaviour ──────────────────────────────────────────────────────

    async def test_5xx_is_retried_and_eventually_succeeds(self, monkeypatch):
        """First attempt returns 503; second attempt returns a valid SSE stream."""
        call_count = 0
        sse_body = b'data: {"text": "hi"}\n\n'

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    503, stream=_AsyncSingleChunkStream(b"down"), request=request
                )
            return httpx.Response(
                200, stream=_AsyncSingleChunkStream(sse_body), request=request
            )

        monkeypatch.setattr(
            upstream.httpx, "AsyncClient", _make_client_factory(handler)
        )
        monkeypatch.setattr(upstream.asyncio, "sleep", AsyncMock())

        texts = [t async for t, _ in upstream.stream_upstream_response("Hello", [])]

        assert texts == ["hi"]
        assert call_count == 2

    async def test_4xx_is_not_retried(self, monkeypatch):
        """A 429 error should be raised immediately without any retry."""
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                429, stream=_AsyncSingleChunkStream(b"rate limit"), request=request
            )

        monkeypatch.setattr(
            upstream.httpx, "AsyncClient", _make_client_factory(handler)
        )

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            async for _ in upstream.stream_upstream_response("Hello", []):
                pass

        assert exc_info.value.response.status_code == 429
        assert call_count == 1  # no retry

    async def test_exhausted_retries_raise_last_exception(self, monkeypatch):
        """All attempts fail with 502 — the final exception must propagate."""
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(
                502, stream=_AsyncSingleChunkStream(b"bad gateway"), request=request
            )

        monkeypatch.setattr(
            upstream.httpx, "AsyncClient", _make_client_factory(handler)
        )
        monkeypatch.setattr(upstream.asyncio, "sleep", AsyncMock())

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            async for _ in upstream.stream_upstream_response("Hello", []):
                pass

        assert exc_info.value.response.status_code == 502
        # Default UPSTREAM_RETRY_ATTEMPTS=3, so exactly 3 calls expected.
        assert call_count == 3

    async def test_network_error_is_retried(self, monkeypatch):
        """A ConnectError on first attempt should trigger a retry."""
        call_count = 0
        sse_body = b'data: {"text": "recovered"}\n\n'

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(
                200, stream=_AsyncSingleChunkStream(sse_body), request=request
            )

        monkeypatch.setattr(
            upstream.httpx, "AsyncClient", _make_client_factory(handler)
        )
        monkeypatch.setattr(upstream.asyncio, "sleep", AsyncMock())

        texts = [t async for t, _ in upstream.stream_upstream_response("Hello", [])]

        assert texts == ["recovered"]
        assert call_count == 2


class TestExtractUsageFromSseData:
    # ── Claude native message_start ──────────────────────────────────────────

    def test_claude_message_start_returns_prompt_and_completion(self):
        data = json.dumps(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "usage": {"input_tokens": 25, "output_tokens": 1},
                },
            }
        )
        assert _extract_usage_from_sse_data(data) == {
            "prompt_tokens": 25,
            "completion_tokens": 1,
        }

    def test_claude_message_start_missing_usage_returns_none(self):
        data = json.dumps({"type": "message_start", "message": {"id": "msg_1"}})
        assert _extract_usage_from_sse_data(data) is None

    # ── Claude native message_delta ──────────────────────────────────────────

    def test_claude_message_delta_returns_completion_tokens(self):
        data = json.dumps(
            {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 15}}
        )
        assert _extract_usage_from_sse_data(data) == {"completion_tokens": 15}

    def test_claude_message_delta_missing_usage_returns_none(self):
        data = json.dumps(
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
        )
        assert _extract_usage_from_sse_data(data) is None

    # ── OpenAI-like usage object ─────────────────────────────────────────────

    def test_openai_like_usage_returns_prompt_and_completion(self):
        data = json.dumps(
            {
                "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 20,
                    "total_tokens": 50,
                },
            }
        )
        assert _extract_usage_from_sse_data(data) == {
            "prompt_tokens": 30,
            "completion_tokens": 20,
        }

    def test_openai_like_no_usage_field_returns_none(self):
        data = json.dumps({"choices": [{"delta": {"content": "hello"}}]})
        assert _extract_usage_from_sse_data(data) is None

    # ── Non-usage events and edge cases ─────────────────────────────────────

    def test_content_block_delta_returns_none(self):
        data = json.dumps(
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "hi"},
            }
        )
        assert _extract_usage_from_sse_data(data) is None

    def test_done_sentinel_returns_none(self):
        assert _extract_usage_from_sse_data("[DONE]") is None

    def test_empty_string_returns_none(self):
        assert _extract_usage_from_sse_data("") is None

    def test_raw_text_returns_none(self):
        assert _extract_usage_from_sse_data("plain text chunk") is None
