"""
Unit tests for utils/converters.py

These are pure logic tests — no HTTP, no FastAPI, no mocking required.
"""

from __future__ import annotations

import json


from schemas.openai import (
    ChatCompletionRequest,
    FunctionCall,
    Message,
    Tool,
    FunctionDefinition,
    ToolCall,
)
from utils.converters import (
    looks_like_json_start,
    make_delta_tool_calls,
    messages_to_upstream_format,
    tool_calls_to_openai,
    try_parse_tool_calls,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def make_request(
    messages: list[Message], tools: list[Tool] | None = None
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="gateway-adapter", messages=messages, tools=tools
    )


# ── messages_to_upstream_format ──────────────────────────────────────────────────────


class TestMessagesToBFormat:
    def test_single_user_message_becomes_query(self):
        req = make_request([Message(role="user", content="Hello!")])
        query, history = messages_to_upstream_format(req)

        assert query == "Hello!"
        # No prior turns, so history only has the system preamble (none here)
        assert not any(h["role"] == "user" for h in history)

    def test_system_message_injected_into_history(self):
        req = make_request(
            [
                Message(role="system", content="You are a pirate."),
                Message(role="user", content="Hi"),
            ]
        )
        query, history = messages_to_upstream_format(req)

        assert query == "Hi"
        system_entries = [h for h in history if h["role"] == "system"]
        assert len(system_entries) == 1
        assert "You are a pirate." in system_entries[0]["content"]

    def test_multiple_system_messages_merged(self):
        req = make_request(
            [
                Message(role="system", content="Part A."),
                Message(role="system", content="Part B."),
                Message(role="user", content="Go"),
            ]
        )
        _, history = messages_to_upstream_format(req)
        combined = next(h["content"] for h in history if h["role"] == "system")
        assert "Part A." in combined and "Part B." in combined

    def test_previous_user_and_assistant_turns_go_to_history(self):
        req = make_request(
            [
                Message(role="user", content="First question"),
                Message(role="assistant", content="First answer"),
                Message(role="user", content="Second question"),
            ]
        )
        query, history = messages_to_upstream_format(req)

        assert query == "Second question"
        user_entries = [h for h in history if h["role"] == "user"]
        assert any("First question" in h["content"] for h in user_entries)
        assistant_entries = [h for h in history if h["role"] == "assistant"]
        assert any("First answer" in h["content"] for h in assistant_entries)

    def test_assistant_tool_calls_serialised_as_json_in_history(self):
        req = make_request(
            [
                Message(role="user", content="What's the weather?"),
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            type="function",
                            function=FunctionCall(
                                name="get_weather",
                                arguments='{"location": "Seoul"}',
                            ),
                        )
                    ],
                ),
                Message(role="tool", tool_call_id="call_1", content='{"temp": 20}'),
                Message(role="user", content="Thanks"),
            ]
        )
        query, history = messages_to_upstream_format(req)

        assert query == "Thanks"

        # The assistant turn should be serialised as JSON
        assistant_entry = next(h for h in history if h["role"] == "assistant")
        data = json.loads(assistant_entry["content"])
        assert data["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_result_converted_to_user_message(self):
        req = make_request(
            [
                Message(role="user", content="Run it"),
                Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="call_2",
                            type="function",
                            function=FunctionCall(name="run", arguments="{}"),
                        )
                    ],
                ),
                Message(role="tool", tool_call_id="call_2", content="output: 42"),
                Message(role="user", content="Now summarise"),
            ]
        )
        _, history = messages_to_upstream_format(req)

        # Tool results are now emitted as a structured JSON object so the
        # format is language-neutral and symmetric with the tool_calls format.
        tool_result_entry = next(
            (
                h
                for h in history
                if h["role"] == "user"
                and '"tool_results"' in h["content"]
                and "call_2" in h["content"]
            ),
            None,
        )
        assert tool_result_entry is not None
        data = json.loads(tool_result_entry["content"])
        assert data["tool_results"][0]["tool_call_id"] == "call_2"
        assert data["tool_results"][0]["content"] == "output: 42"

    def test_no_user_message_returns_empty_query(self):
        req = make_request([Message(role="assistant", content="Hello")])
        query, _ = messages_to_upstream_format(req)
        assert query == ""

    def test_tools_inject_system_prompt(self):
        tool = Tool(
            type="function",
            function=FunctionDefinition(name="my_tool", description="Does stuff"),
        )
        req = make_request([Message(role="user", content="Go")], tools=[tool])
        _, history = messages_to_upstream_format(req)

        system_entry = next(h["content"] for h in history if h["role"] == "system")
        assert "my_tool" in system_entry
        assert "tool_calls" in system_entry  # the JSON format example

    def test_tools_system_prompt_includes_original_system(self):
        tool = Tool(type="function", function=FunctionDefinition(name="t"))
        req = make_request(
            [
                Message(role="system", content="Custom instruction."),
                Message(role="user", content="Go"),
            ],
            tools=[tool],
        )
        _, history = messages_to_upstream_format(req)

        sys_content = next(h["content"] for h in history if h["role"] == "system")
        assert "Custom instruction." in sys_content


# ── try_parse_tool_calls ──────────────────────────────────────────────────────


class TestTryParseToolCalls:
    def test_valid_tool_call_json(self):
        text = json.dumps(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                    }
                ]
            }
        )
        result = try_parse_tool_calls(text)
        assert result is not None
        assert result[0]["function"]["name"] == "add"

    def test_strips_markdown_json_fences(self):
        text = '```json\n{"tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}\n```'
        result = try_parse_tool_calls(text)
        assert result is not None
        assert result[0]["function"]["name"] == "f"

    def test_strips_plain_markdown_fences(self):
        text = '```\n{"tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}\n```'
        result = try_parse_tool_calls(text)
        assert result is not None

    def test_plain_text_returns_none(self):
        assert try_parse_tool_calls("Hello world") is None

    def test_invalid_json_returns_none(self):
        assert try_parse_tool_calls("{broken json") is None

    def test_json_without_tool_calls_key_returns_none(self):
        assert try_parse_tool_calls('{"answer": 42}') is None

    def test_whitespace_around_json_handled(self):
        text = '  \n  {"tool_calls": [{"id": "c", "type": "function", "function": {"name": "g", "arguments": "{}"}}]}  \n  '
        result = try_parse_tool_calls(text)
        assert result is not None

    def test_multiple_tool_calls(self):
        text = json.dumps(
            {
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "a", "arguments": "{}"},
                    },
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "b", "arguments": "{}"},
                    },
                ]
            }
        )
        result = try_parse_tool_calls(text)
        assert result is not None
        assert len(result) == 2
        assert result[1]["function"]["name"] == "b"


# ── looks_like_json_start ─────────────────────────────────────────────────────


class TestLooksLikeJsonStart:
    def test_opening_brace_returns_true(self):
        assert looks_like_json_start("{") is True

    def test_leading_whitespace_with_brace_returns_true(self):
        assert looks_like_json_start("  \n{") is True

    def test_plain_text_returns_false(self):
        assert looks_like_json_start("Hello") is False

    def test_empty_string_returns_false(self):
        assert looks_like_json_start("") is False

    def test_array_start_returns_true(self):
        assert looks_like_json_start("[1,2,3]") is True

    def test_leading_whitespace_with_array_returns_true(self):
        assert looks_like_json_start("  \n[") is True


# ── make_delta_tool_calls ─────────────────────────────────────────────────────


class TestMakeDeltaToolCalls:
    def _make_raw(self, name="my_fn", args='{"x": 1}', call_id="call_abc"):
        return [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ]

    def test_phase1_contains_id_and_name(self):
        phase1, _ = make_delta_tool_calls(self._make_raw())
        assert phase1[0].function is not None
        assert phase1[0].id == "call_abc"
        assert phase1[0].function.name == "my_fn"

    def test_phase1_arguments_are_empty(self):
        phase1, _ = make_delta_tool_calls(self._make_raw())
        assert phase1[0].function is not None
        assert phase1[0].function.arguments == ""

    def test_phase2_contains_arguments(self):
        _, phase2 = make_delta_tool_calls(self._make_raw(args='{"x": 99}'))
        assert phase2[0].function is not None
        assert phase2[0].function.arguments == '{"x": 99}'

    def test_phase2_has_no_id_or_name(self):
        _, phase2 = make_delta_tool_calls(self._make_raw())
        assert phase2[0].function is not None
        assert phase2[0].id is None
        assert phase2[0].function.name is None

    def test_multiple_tools_indexed_correctly(self):
        raw = [
            {
                "id": "c0",
                "type": "function",
                "function": {"name": "fn0", "arguments": "{}"},
            },
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "fn1", "arguments": "{}"},
            },
        ]
        phase1, phase2 = make_delta_tool_calls(raw)
        assert phase1[0].index == 0
        assert phase1[1].index == 1
        assert phase2[0].index == 0
        assert phase2[1].index == 1


# ── tool_calls_to_openai ──────────────────────────────────────────────────────


class TestToolCallsToOpenai:
    def test_basic_conversion(self):
        raw = [
            {
                "id": "call_x",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "hi"}'},
            }
        ]
        result = tool_calls_to_openai(raw)
        assert len(result) == 1
        assert result[0].id == "call_x"
        assert result[0].function.name == "search"
        assert result[0].function.arguments == '{"q": "hi"}'

    def test_missing_id_gets_fallback(self):
        raw = [{"type": "function", "function": {"name": "fn", "arguments": "{}"}}]
        result = tool_calls_to_openai(raw)
        assert result[0].id.startswith("call_")

    def test_missing_arguments_defaults_to_empty_object(self):
        raw = [{"id": "c", "type": "function", "function": {"name": "fn"}}]
        result = tool_calls_to_openai(raw)
        assert result[0].function.arguments == "{}"
