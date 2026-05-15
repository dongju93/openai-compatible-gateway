"""
Bidirectional conversion helpers between:
  • OpenAI Chat Completions message format
  • The upstream API's query / history wire format

Conversion summary
------------------
OpenAI side          →   upstream side
─────────────────────────────────────────────────
system message       →   injected as first history entry (role="system")
previous user msgs   →   history entries  (role="user")
assistant (text)     →   history entries  (role="assistant")
assistant (tool_call)→   history entry    (role="assistant", content=JSON repr)
tool result          →   history entry    (role="user",      content=labelled str)
last user message    →   query  (top-level field)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from schemas.openai import (
    ChatCompletionRequest,
    DeltaToolCall,
    DeltaToolCallFunction,
    FunctionCall,
    ToolCall,
)
from utils.prompt_injector import build_tool_system_prompt

logger = logging.getLogger(__name__)


# ── OpenAI messages → upstream (query, history) ──────────────────────────────


def messages_to_upstream_format(
    request: ChatCompletionRequest,
) -> tuple[str, list[dict]]:
    """
    Convert an OpenAI ChatCompletionRequest into the ``(query, history)`` pair
    expected by the upstream API.

    History entry schema:
        {"role": "user" | "assistant" | "system", "content": str}

    Returns:
        query:   The text of the last user message in the request.
        history: All preceding turns plus the system/tool-injection preamble.
    """
    messages = request.messages
    tools = request.tools or []

    # ── 1. Collect system-level content from all system messages ──────────────
    system_parts: list[str] = []
    non_system = []
    for msg in messages:
        if msg.role == "system":
            if msg.content:
                system_parts.append(msg.content)
        else:
            non_system.append(msg)

    original_system = "\n\n".join(system_parts) if system_parts else None

    # ── 2. Build the effective system prompt ──────────────────────────────────
    # If tools are present, the tool-injection prompt is merged here.
    if tools:
        effective_system = build_tool_system_prompt(tools, extra_system=original_system)
    else:
        effective_system = original_system

    # ── 3. Identify the last user message — it becomes the query ─────────────
    last_user_idx = next(
        (i for i in range(len(non_system) - 1, -1, -1) if non_system[i].role == "user"),
        -1,
    )

    if last_user_idx == -1:
        query = ""
        history_messages = non_system
    else:
        query = non_system[last_user_idx].content or ""
        history_messages = non_system[:last_user_idx]

    # ── 4. Build the history list ─────────────────────────────────────────────
    history: list[dict] = []

    # System/tool-injection prompt goes first so upstream always sees it
    if effective_system:
        history.append({"role": "system", "content": effective_system})

    for msg in history_messages:
        if msg.role == "user":
            history.append({"role": "user", "content": msg.content or ""})

        elif msg.role == "assistant":
            if msg.tool_calls:
                # Serialise tool calls as the exact JSON the model would have emitted.
                # This keeps the history format consistent with what the
                # tool-injection prompt describes when upstream sees prior turns.
                tc_payload = {
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                }
                history.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(tc_payload, ensure_ascii=False),
                    }
                )
            else:
                history.append({"role": "assistant", "content": msg.content or ""})

        elif msg.role == "tool":
            # Tool results are folded into the user turn using the same JSON
            # protocol taught in the system prompt.  A structured object avoids
            # relying on English prose that non-English-trained models may not
            # associate with the preceding tool call.
            history.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "tool_results": [
                                {
                                    "tool_call_id": msg.tool_call_id,
                                    "content": msg.content or "",
                                }
                            ]
                        },
                        ensure_ascii=False,
                    ),
                }
            )

    return query, history


# ── Tool-call JSON detection ──────────────────────────────────────────────────

_JSON_START_RE = re.compile(r"^\s*[{\[]")


def looks_like_json_start(text: str) -> bool:
    """Return True if the text appears to be the start of a JSON value (object or array)."""
    return bool(_JSON_START_RE.match(text))


def try_parse_tool_calls(text: str) -> Optional[list[dict]]:
    """
    Attempt to extract the ``tool_calls`` list from raw model output.

    The model is instructed to emit exactly one JSON object containing a
    ``tool_calls`` key when invoking tools.  This function:
      1. Strips optional markdown code fences (the model sometimes adds them
         even when told not to).
      2. JSON-parses the result.
      3. Validates the expected shape.

    Returns:
        A list of raw tool-call dicts on success, or None if the text is not
        a recognised tool-call payload.
    """
    cleaned = text.strip()

    # Strip ``` ... ``` wrappers the model may add despite instructions
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    if not cleaned.startswith("{"):
        return None

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.debug(
            "Could not parse potential tool-call JSON (first 200 chars): %s",
            cleaned[:200],
        )
        return None

    if isinstance(data, dict) and isinstance(data.get("tool_calls"), list):
        return data["tool_calls"]

    return None


# ── Raw tool-call dicts → OpenAI schema objects ───────────────────────────────


def tool_calls_to_openai(raw_tool_calls: list[dict]) -> list[ToolCall]:
    """Convert raw parsed dicts into typed ToolCall schema objects (non-streaming)."""
    result = []
    for i, tc in enumerate(raw_tool_calls):
        func = tc.get("function", {})
        result.append(
            ToolCall(
                id=tc.get("id", f"call_{i:04d}"),
                type="function",
                function=FunctionCall(
                    name=func.get("name", ""),
                    arguments=func.get("arguments", "{}"),
                ),
            )
        )
    return result


def make_delta_tool_calls(
    raw_tool_calls: list[dict],
) -> tuple[list[DeltaToolCall], list[DeltaToolCall]]:
    """
    Build the two-phase ``delta.tool_calls`` lists for streaming.

    OpenAI's streaming spec requires tool calls emitted in two separate chunks:

    Phase 1 — announces the tool call (index, id, type, function.name, empty args):
        {"index": 0, "id": "call_abc", "type": "function",
         "function": {"name": "get_weather", "arguments": ""}}

    Phase 2 — delivers the arguments:
        {"index": 0, "function": {"arguments": "{\"location\": \"NYC\"}"}}

    Separating them is important because @ai-sdk/openai-compatible uses the
    phase-1 chunk to create the tool-call slot and phase-2 to fill it.
    """
    phase1: list[DeltaToolCall] = []
    phase2: list[DeltaToolCall] = []

    for i, tc in enumerate(raw_tool_calls):
        func = tc.get("function", {})
        phase1.append(
            DeltaToolCall(
                index=i,
                id=tc.get("id", f"call_{i:04d}"),
                type="function",
                function=DeltaToolCallFunction(
                    name=func.get("name", ""),
                    arguments="",  # empty in phase 1
                ),
            )
        )
        phase2.append(
            DeltaToolCall(
                index=i,
                # id/type omitted in phase 2 — client already knows them
                function=DeltaToolCallFunction(
                    arguments=func.get("arguments", "{}"),
                ),
            )
        )

    return phase1, phase2
