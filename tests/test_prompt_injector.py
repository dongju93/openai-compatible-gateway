"""
Unit tests for utils/prompt_injector.py

Verifies that the tool-injection system prompt is well-formed and contains
all the content that the model needs to know about its tools.
"""

from __future__ import annotations

import json


from schemas.openai import FunctionDefinition, Tool
from utils.prompt_injector import build_tool_system_prompt


def _make_tool(name: str, description: str = "", params: dict | None = None) -> Tool:
    return Tool(
        type="function",
        function=FunctionDefinition(
            name=name,
            description=description or f"Does {name}",
            parameters=params or {"type": "object", "properties": {}},
        ),
    )


class TestBuildToolSystemPrompt:
    def test_contains_tool_name(self):
        prompt = build_tool_system_prompt([_make_tool("search_web")])
        assert "search_web" in prompt

    def test_contains_all_tool_names_for_multiple_tools(self):
        tools = [
            _make_tool("read_file"),
            _make_tool("write_file"),
            _make_tool("delete_file"),
        ]
        prompt = build_tool_system_prompt(tools)
        assert "read_file" in prompt
        assert "write_file" in prompt
        assert "delete_file" in prompt

    def test_contains_tool_call_json_format_example(self):
        """The prompt must include the JSON format the model must use."""
        prompt = build_tool_system_prompt([_make_tool("fn")])
        assert "tool_calls" in prompt
        assert '"function"' in prompt or "function" in prompt

    def test_tools_are_valid_json_in_prompt(self):
        """The serialised tools block inside the prompt must be valid JSON."""
        tools = [
            _make_tool(
                "calc",
                "Calculates things",
                {"type": "object", "properties": {"expr": {"type": "string"}}},
            )
        ]
        prompt = build_tool_system_prompt(tools)

        # Extract the JSON block between "## Available Tools\n\n" and "## How to Call"
        start = prompt.index("## Available Tools") + len("## Available Tools")
        end = prompt.index("## How to Call a Tool")
        json_block = prompt[start:end].strip()
        parsed = json.loads(json_block)  # raises if not valid JSON
        assert parsed[0]["function"]["name"] == "calc"

    def test_extra_system_content_prepended(self):
        prompt = build_tool_system_prompt(
            [_make_tool("fn")], extra_system="Be concise."
        )
        # Extra system content should appear before the tool rules
        concise_pos = prompt.index("Be concise.")
        tool_pos = prompt.index("Available Tools")
        assert concise_pos < tool_pos

    def test_no_extra_system_returns_base_prompt_only(self):
        prompt = build_tool_system_prompt([_make_tool("fn")], extra_system=None)
        assert "Be concise." not in prompt

    def test_prompt_contains_critical_rules(self):
        """The model must be told to output ONLY JSON when calling a tool."""
        prompt = build_tool_system_prompt([_make_tool("fn")])
        # At least one of these phrases must be present
        critical_phrases = ["ONLY", "only", "nothing else", "no other text"]
        assert any(p in prompt for p in critical_phrases)

    def test_tool_description_included(self):
        prompt = build_tool_system_prompt(
            [_make_tool("magic_fn", "Performs magic operations")]
        )
        assert "Performs magic operations" in prompt
