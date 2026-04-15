"""
Builds the system-prompt injection that teaches the upstream model how to call tools.

Why prompt injection?
---------------------
The upstream API has no native tool-calling surface — it only accepts `query`
and `history`.  To give the model tool-awareness we:

  1. Serialize the entire OpenAI tools array into this system prompt.
  2. Instruct the model on the EXACT JSON format it must use when calling a tool.
  3. Parse that JSON back into OpenAI-format tool_calls in the gateway.

The format is chosen to be unambiguous: the entire response must be a single
JSON object starting with `{` when a tool is needed, so the gateway can detect
it reliably without complex heuristics.
"""

from __future__ import annotations

import json
from typing import Optional

from schemas.openai import Tool


# Single-brace literal braces must be doubled inside an f-string.
_TOOL_PROMPT_TEMPLATE = """\
You are a helpful coding agent that can use tools to complete tasks.

## Available Tools

{tools_json}

## How to Call a Tool

When you need to call one or more tools, output ONLY the following JSON object.
Do NOT include any text before or after it.
Do NOT wrap it in markdown code fences (no ```json ... ```).
Do NOT explain that you are calling a tool.

{{
  "tool_calls": [
    {{
      "id": "call_<unique_alphanumeric_id>",
      "type": "function",
      "function": {{
        "name": "<tool_name_exactly_as_listed_above>",
        "arguments": "<valid JSON string with escaped inner quotes>"
      }}
    }}
  ]
}}

## Critical Rules

1. When calling a tool → output ONLY the JSON object above, nothing else.
2. You may call multiple tools in one response by including multiple entries
   in the `tool_calls` array.
3. `arguments` MUST be a JSON-encoded *string* — inner quotes must be escaped
   (e.g. `"{{\\"key\\": \\"value\\"}}"` not a raw object).
4. Use a unique, short alphanumeric ID for each call, e.g. "call_a1b2c3".
5. After outputting the JSON, STOP immediately. No follow-up text.
6. If you do NOT need a tool, respond with plain conversational text as usual.
"""


def build_tool_system_prompt(
    tools: list[Tool],
    extra_system: Optional[str] = None,
) -> str:
    """
    Generate the system prompt that encodes tool definitions and calling rules.

    Args:
        tools:        Tool list from the incoming OpenAI request.
        extra_system: Any system-message content already present in the request;
                      prepended so user-supplied instructions take precedence.

    Returns:
        A complete system-prompt string ready to inject at the start of history.
    """
    tools_json = json.dumps(
        [tool.model_dump(exclude_none=True) for tool in tools],
        indent=2,
        ensure_ascii=False,
    )
    prompt = _TOOL_PROMPT_TEMPLATE.format(tools_json=tools_json)

    if extra_system:
        # Prepend the user-supplied system content so it takes priority
        prompt = f"{extra_system}\n\n---\n\n{prompt}"

    return prompt
