"""
Unit tests for services/upstream.py

Tests focus on _extract_text_from_sse_data — the multi-format SSE parser —
since that is the only pure logic in upstream (the network I/O is tested
indirectly via the router integration tests).
"""

from __future__ import annotations

import json


from services.upstream import _extract_text_from_sse_data


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
