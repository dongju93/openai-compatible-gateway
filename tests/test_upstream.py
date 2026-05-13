"""
Unit tests for services/upstream.py

Tests primarily focus on _extract_text_from_sse_data — the multi-format SSE
parser — plus narrow HTTP-client edge cases around streaming response handling.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

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


class TestStreamUpstreamResponse:
    async def test_http_error_response_body_is_read_before_raise(self, monkeypatch):
        real_async_client = httpx.AsyncClient

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                stream=_AsyncSingleChunkStream(b"quota exceeded"),
                request=request,
            )

        def make_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_async_client(*args, **kwargs)

        monkeypatch.setattr(upstream.httpx, "AsyncClient", make_client)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            async for _text, _usage in upstream.stream_upstream_response("Hello", []):
                pass

        assert exc_info.value.response.status_code == 429
        assert exc_info.value.response.text == "quota exceeded"


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
