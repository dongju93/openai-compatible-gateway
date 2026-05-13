"""
Async HTTP client for the upstream chat API.

Responsibilities
----------------
* POST query + history to the upstream chat endpoint using httpx.
* Parse the SSE stream and yield plain text chunks.
* Handle multiple common SSE payload shapes so the gateway is resilient to
  minor variations in the upstream response format.

Supported SSE data formats (tried in order)
--------------------------------------------
1. Claude native:
       {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "..."}}
2. Simple delta:
       {"delta": {"text": "..."}}
3. Flat text:
       {"text": "..."}
4. OpenAI-like passthrough:
       {"choices": [{"delta": {"content": "..."}}]}
5. Raw string (no JSON wrapper):
       Hello, world
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Optional

import httpx

from config import get_settings

logger = logging.getLogger(__name__)

# Type alias: each yielded event is either (text, None) or (None, usage_dict)
UpstreamEvent = tuple[Optional[str], Optional[dict[str, int]]]


def _extract_text_from_sse_data(data: str) -> Optional[str]:
    """
    Extract a text fragment from a single SSE ``data:`` payload.

    Returns the extracted text string, or None if the payload carries no text
    (e.g. it is a metadata event like ``message_start`` or ``[DONE]``).
    """
    if not data or data.strip() == "[DONE]":
        return None

    # ── Attempt JSON parsing ──────────────────────────────────────────────────
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        # Not JSON → treat the raw data as a text chunk
        stripped = data.strip()
        return stripped if stripped else None

    if not isinstance(obj, dict):
        return None

    # 1. Claude native streaming: content_block_delta / text_delta
    if obj.get("type") == "content_block_delta":
        delta = obj.get("delta", {})
        if delta.get("type") == "text_delta":
            return delta.get("text") or None

    # 2. {"delta": {"text": "..."}}
    delta = obj.get("delta")
    if isinstance(delta, dict) and "text" in delta:
        return delta["text"] or None

    # 3. {"text": "..."}
    if "text" in obj:
        return obj["text"] or None

    # 4. OpenAI-like: {"choices": [{"delta": {"content": "..."}}]}
    choices = obj.get("choices", [])
    if choices and isinstance(choices, list):
        content = choices[0].get("delta", {}).get("content")
        return content or None

    return None


def _extract_usage_from_sse_data(data: str) -> Optional[dict[str, int]]:
    """
    Extract token-usage data from a single SSE ``data:`` payload.

    Returns a partial dict with ``prompt_tokens`` and/or ``completion_tokens``,
    or None if the payload carries no usage information.

    Handled formats
    ---------------
    • Claude native ``message_start``:
          {"type": "message_start", "message": {"usage": {"input_tokens": N, "output_tokens": K}}}
      → ``prompt_tokens=N, completion_tokens=K``

    • Claude native ``message_delta`` (final output token count):
          {"type": "message_delta", "usage": {"output_tokens": M}}
      → ``completion_tokens=M``  (overwrites the K from message_start)

    • OpenAI-like final chunk:
          {"usage": {"prompt_tokens": N, "completion_tokens": M}}
      → ``prompt_tokens=N, completion_tokens=M``
    """
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    event_type = obj.get("type")

    # Claude native: message_start carries input (prompt) token count
    if event_type == "message_start":
        msg_usage = obj.get("message", {}).get("usage", {})
        if msg_usage:
            return {
                "prompt_tokens": int(msg_usage.get("input_tokens", 0)),
                "completion_tokens": int(msg_usage.get("output_tokens", 0)),
            }

    # Claude native: message_delta carries the definitive output token count
    if event_type == "message_delta":
        delta_usage = obj.get("usage", {})
        if delta_usage and "output_tokens" in delta_usage:
            return {"completion_tokens": int(delta_usage["output_tokens"])}

    # OpenAI-like: top-level "usage" object (typically in the final streaming chunk)
    usage_obj = obj.get("usage")
    if isinstance(usage_obj, dict) and (
        "prompt_tokens" in usage_obj or "completion_tokens" in usage_obj
    ):
        return {
            "prompt_tokens": int(usage_obj.get("prompt_tokens", 0)),
            "completion_tokens": int(usage_obj.get("completion_tokens", 0)),
        }

    return None


async def stream_upstream_response(
    query: str,
    history: list[dict],
    api_key_override: Optional[str] = None,
    generation_params: Optional[dict] = None,
) -> AsyncIterator[UpstreamEvent]:
    """
    Call the upstream API (streaming) and yield events as ``(text, usage)`` tuples.

    Each yielded tuple has exactly one non-None field:
    • ``(text, None)``  — an incremental text fragment
    • ``(None, usage)`` — emitted once after the stream ends, only when the
                          upstream provided token-count data; ``usage`` is a dict
                          with ``prompt_tokens`` and ``completion_tokens`` keys

    The gateway calls this for *both* streaming and non-streaming OpenAI
    requests — for non-streaming we simply collect all text fragments.

    Args:
        query:            The current user message (last in conversation).
        history:          Conversation history in the upstream wire format.
        api_key_override: Per-request API key; falls back to settings.UPSTREAM_API_KEY.

    Yields:
        UpstreamEvent: ``(text_fragment, None)`` or ``(None, usage_dict)``

    Raises:
        httpx.HTTPStatusError: Propagated if upstream returns a 4xx / 5xx status.
        httpx.RequestError:    Propagated on network-level failures.
    """
    settings = get_settings()
    url = f"{settings.UPSTREAM_BASE_URL}{settings.UPSTREAM_CHAT_ENDPOINT}"

    effective_key = api_key_override or settings.UPSTREAM_API_KEY
    headers: dict[str, str] = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }
    if effective_key:
        headers["Authorization"] = f"Bearer {effective_key}"

    payload: dict = {"query": query, "history": json.dumps(history), "stream": True}
    if generation_params:
        payload.update(generation_params)

    logger.debug(
        "Calling upstream  url=%s  query_len=%d  history_len=%d",
        url,
        len(query),
        len(history),
    )

    usage_accum: dict[str, int] = {}

    async with httpx.AsyncClient(timeout=settings.UPSTREAM_TIMEOUT) as client:
        async with client.stream(
            "POST", url, json=payload, headers=headers
        ) as response:
            if not response.is_success:
                # Streaming responses do not expose .text unless the body is read.
                await response.aread()
            response.raise_for_status()

            async for raw_line in response.aiter_lines():
                raw_line = raw_line.strip()

                if not raw_line:
                    continue  # blank lines are SSE heartbeats / separators

                if raw_line.startswith("data:"):
                    # Standard SSE — strip the "data:" prefix (with optional space)
                    data = raw_line[5:].lstrip(" ")
                    text = _extract_text_from_sse_data(data)
                    if text:
                        yield (text, None)
                    usage_fragment = _extract_usage_from_sse_data(data)
                    if usage_fragment:
                        usage_accum.update(usage_fragment)

                elif raw_line.startswith("event:") or raw_line.startswith("id:"):
                    # Event-type / id lines — no text content, skip
                    continue

                else:
                    # Some lightweight implementations omit the "data:" prefix
                    text = _extract_text_from_sse_data(raw_line)
                    if text:
                        yield (text, None)
                    usage_fragment = _extract_usage_from_sse_data(raw_line)
                    if usage_fragment:
                        usage_accum.update(usage_fragment)

    # Emit accumulated usage once after the stream is fully consumed, only if
    # the upstream actually provided token counts.
    if usage_accum:
        yield (None, usage_accum)
