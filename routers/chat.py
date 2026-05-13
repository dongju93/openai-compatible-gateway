"""
POST /v1/chat/completions — OpenAI-compatible chat completions endpoint.

This router is the heart of the gateway.  For every request it:

  1. Converts the OpenAI message list to the upstream (query, history) format.
     Tool definitions are injected into the system prompt at this step.
  2. Calls the upstream API and receives a streaming text response.
  3. Detects whether the response is a tool-call JSON block or plain text.
  4. Returns either:
       • A streaming ``text/event-stream`` response (stream=True)
       • A complete JSON response              (stream=False)
     …both shaped identically to what the real OpenAI API returns.

Streaming detection strategy
-----------------------------
We buffer the incoming stream until we see the first non-whitespace character.
If that character is ``{`` we switch to full JSON-buffering mode (tool call
likely).  Otherwise we start forwarding content chunks immediately (plain text
response).  This gives us:
  • Zero-latency streaming for normal text replies.
  • Correct, complete JSON for tool-call parsing.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import AsyncIterator, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from schemas.openai import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Delta,
    StreamChoice,
    Usage,
)
from services.upstream import stream_upstream_response
from utils.converters import (
    looks_like_json_start,
    make_delta_tool_calls,
    messages_to_upstream_format,
    tool_calls_to_openai,
    try_parse_tool_calls,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Small helpers ─────────────────────────────────────────────────────────────

_FORWARDED_PARAMS = (
    "temperature",
    "max_tokens",
    "top_p",
    "stop",
    "presence_penalty",
    "frequency_penalty",
)


def _generation_params(request: ChatCompletionRequest) -> dict:
    """Extract generation parameters to forward to the upstream payload."""
    return {
        key: value
        for key in _FORWARDED_PARAMS
        if (value := getattr(request, key, None)) is not None
    }


def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _sse(chunk: ChatCompletionChunk) -> str:
    """Serialize a chunk to a single SSE data line (with trailing newlines)."""
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


def _build_usage(accum: dict[str, int]) -> Optional[Usage]:
    """Convert an upstream usage accumulator to a Usage object, or None if empty."""
    if not accum:
        return None
    prompt = accum.get("prompt_tokens", 0)
    completion = accum.get("completion_tokens", 0)
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _build_chunk(
    completion_id: str,
    model: str,
    delta: Delta,
    finish_reason: Optional[str] = None,
    usage: Optional[Usage] = None,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=completion_id,
        created=int(time.time()),
        model=model,
        choices=[StreamChoice(delta=delta, finish_reason=finish_reason)],
        usage=usage,
    )


# ── Streaming generator ───────────────────────────────────────────────────────


async def _stream_response(
    request: ChatCompletionRequest,
    completion_id: str,
) -> AsyncIterator[str]:
    """
    Core streaming generator — yields SSE-formatted strings.

    Phases:
      detection  — buffer until we can tell text vs JSON
      content    — forward chunks immediately as delta.content
      json       — buffer the full JSON, then emit as delta.tool_calls
    """
    query, history = messages_to_upstream_format(request)

    # OpenAI SDK expects the very first chunk to carry role="assistant"
    yield _sse(_build_chunk(completion_id, request.model, Delta(role="assistant")))

    buffer = ""
    mode: Optional[str] = None  # None=detecting  "content"=streaming  "json"=buffering
    usage_accum: dict[str, int] = {}

    try:
        async for raw_chunk, usage_data in stream_upstream_response(
            query, history, generation_params=_generation_params(request)
        ):
            if usage_data is not None:
                usage_accum.update(usage_data)

            if raw_chunk is None:
                continue

            buffer += raw_chunk

            if mode is None:
                # ── Detection: wait for the first visible character ────────────
                stripped = buffer.lstrip()
                if not stripped:
                    continue  # still only whitespace

                if looks_like_json_start(stripped):
                    mode = "json"
                    # Keep buffering; don't emit yet
                else:
                    mode = "content"
                    # Flush everything accumulated so far
                    yield _sse(
                        _build_chunk(
                            completion_id, request.model, Delta(content=buffer)
                        )
                    )
                    buffer = ""

            elif mode == "content":
                # ── Content mode: forward immediately ─────────────────────────
                yield _sse(
                    _build_chunk(completion_id, request.model, Delta(content=raw_chunk))
                )
                buffer = ""

            # In json mode: just keep accumulating (no yield)

    except httpx.HTTPStatusError as exc:
        logger.error("Upstream returned HTTP error: %s", exc)
        yield _sse(
            _build_chunk(
                completion_id,
                request.model,
                Delta(
                    content=f"\n[Gateway error: upstream returned {exc.response.status_code}]"
                ),
                finish_reason="stop",
            )
        )
        yield "data: [DONE]\n\n"
        return

    except httpx.RequestError as exc:
        logger.error("Network error calling upstream: %s", exc)
        yield _sse(
            _build_chunk(
                completion_id,
                request.model,
                Delta(content=f"\n[Gateway error: could not reach upstream — {exc}]"),
                finish_reason="stop",
            )
        )
        yield "data: [DONE]\n\n"
        return

    # ── End of upstream stream ─────────────────────────────────────────────────
    finish_reason = "stop"

    if mode == "json" and buffer.strip():
        raw_tool_calls = try_parse_tool_calls(buffer)

        if raw_tool_calls:
            finish_reason = "tool_calls"
            phase1, phase2 = make_delta_tool_calls(raw_tool_calls)

            # Phase 1: announce each tool call (name + empty arguments)
            yield _sse(
                _build_chunk(completion_id, request.model, Delta(tool_calls=phase1))
            )
            # Phase 2: deliver the arguments
            yield _sse(
                _build_chunk(completion_id, request.model, Delta(tool_calls=phase2))
            )

        else:
            # Looked like JSON but wasn't a valid tool call → emit as content
            logger.debug("JSON buffer did not contain tool_calls, emitting as content")
            yield _sse(
                _build_chunk(completion_id, request.model, Delta(content=buffer))
            )

    elif mode is None and buffer.strip():
        # Response was entirely whitespace up until EOF (unusual)
        yield _sse(_build_chunk(completion_id, request.model, Delta(content=buffer)))

    # Terminal finish chunk — signals end of turn to the client.
    # usage is None when upstream provides no token counts (clients see null, not zeros).
    yield _sse(
        _build_chunk(
            completion_id,
            request.model,
            Delta(),
            finish_reason=finish_reason,
            usage=_build_usage(usage_accum),
        )
    )
    yield "data: [DONE]\n\n"


# ── Non-streaming path ────────────────────────────────────────────────────────


async def _complete_response(
    request: ChatCompletionRequest,
    completion_id: str,
) -> ChatCompletionResponse:
    """
    Collect the full upstream response and return a single ChatCompletionResponse.

    Internally still uses the streaming endpoint so there is only one HTTP code path.
    """
    query, history = messages_to_upstream_format(request)

    full_text = ""
    usage_accum: dict[str, int] = {}
    async for chunk, usage_data in stream_upstream_response(
        query, history, generation_params=_generation_params(request)
    ):
        if usage_data is not None:
            usage_accum.update(usage_data)
        if chunk is not None:
            full_text += chunk

    raw_tool_calls = try_parse_tool_calls(full_text)

    if raw_tool_calls:
        tool_calls = tool_calls_to_openai(raw_tool_calls)
        message = ChoiceMessage(role="assistant", tool_calls=tool_calls)
        finish_reason = "tool_calls"
    else:
        message = ChoiceMessage(role="assistant", content=full_text)
        finish_reason = "stop"

    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=request.model,
        choices=[Choice(message=message, finish_reason=finish_reason)],
        usage=_build_usage(usage_accum),
    )


# ── Route ─────────────────────────────────────────────────────────────────────


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions.

    Accepts the full OpenAI request schema (including tools, tool_choice,
    streaming flags) and returns properly shaped OpenAI responses.
    Tool calling is implemented via prompt injection into the upstream API.
    """
    if body.n is not None and body.n != 1:
        raise HTTPException(
            status_code=422,
            detail="Parameter 'n' must be 1; this gateway does not support multiple completions per request.",
        )

    completion_id = _new_id()
    logger.info(
        "chat_completions  id=%s  model=%s  stream=%s  messages=%d  tools=%d",
        completion_id,
        body.model,
        body.stream,
        len(body.messages),
        len(body.tools or []),
    )

    if body.stream:
        return StreamingResponse(
            _stream_response(body, completion_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Tells nginx / any reverse proxy NOT to buffer the stream
                "X-Accel-Buffering": "no",
            },
        )

    try:
        response = await _complete_response(body, completion_id)
        return response
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail=f"Upstream error: {exc.response.text}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach upstream: {exc}")
