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


async def stream_upstream_response(
    query: str,
    history: list[dict],
    api_key_override: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Call the upstream API (streaming) and yield raw text chunks as they arrive.

    The gateway calls this for *both* streaming and non-streaming OpenAI
    requests — for non-streaming we simply collect all yielded chunks.

    Args:
        query:            The current user message (last in conversation).
        history:          Conversation history in the upstream wire format.
        api_key_override: Per-request API key; falls back to settings.UPSTREAM_API_KEY.

    Yields:
        str: Incremental text fragments from the upstream response.

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

    logger.debug(
        "Calling upstream  url=%s  query_len=%d  history_len=%d",
        url,
        len(query),
        len(history),
    )

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
                        yield text

                elif raw_line.startswith("event:") or raw_line.startswith("id:"):
                    # Event-type / id lines — no text content, skip
                    continue

                else:
                    # Some lightweight implementations omit the "data:" prefix
                    text = _extract_text_from_sse_data(raw_line)
                    if text:
                        yield text
