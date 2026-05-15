"""
Pydantic v2 schemas that mirror the OpenAI Chat Completions API surface.

These are intentionally kept as close to the real OpenAI spec as possible
so that @ai-sdk/openai-compatible (and tools like opencode) work without
any client-side adaptation.

Key types
---------
ChatCompletionRequest  — incoming request body
ChatCompletionResponse — non-streaming response body
ChatCompletionChunk    — one SSE chunk in a streaming response
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Tool / function definitions ───────────────────────────────────────────────


class FunctionDefinition(BaseModel):
    """JSON-Schema description of a callable function."""

    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


class Tool(BaseModel):
    """A tool the model may call (currently only 'function' type)."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    """The function being invoked inside a ToolCall."""

    name: str
    # OpenAI always encodes arguments as a *JSON string* (not an object)
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


# ── Message types ─────────────────────────────────────────────────────────────


class Message(BaseModel):
    """A single turn in the conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    # Plain text OR array of content-part objects (OpenAI multimodal format).
    # May be None for assistant turns that only have tool_calls.
    content: Optional[Union[str, list[dict[str, Any]]]] = None
    # Present on assistant turns where the model calls one or more tools
    tool_calls: Optional[list[ToolCall]] = None
    # Present on tool-result turns (role == "tool")
    tool_call_id: Optional[str] = None
    # Optional display name
    name: Optional[str] = None


# ── Request helpers ───────────────────────────────────────────────────────────


class StreamOptions(BaseModel):
    include_usage: Optional[bool] = None


class ResponseFormat(BaseModel):
    type: str
    json_schema: Optional[dict[str, Any]] = None
    name: Optional[str] = None
    description: Optional[str] = None
    strict: Optional[bool] = None


# ── Request ───────────────────────────────────────────────────────────────────


class ChatCompletionRequest(BaseModel):
    """Full OpenAI-compatible chat completion request."""

    model: str
    messages: list[Message]
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    response_format: Optional[ResponseFormat] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stop: Optional[Union[str, list[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    parallel_tool_calls: Optional[bool] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None


# ── Non-streaming response ────────────────────────────────────────────────────


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    """The complete message inside a non-streaming choice."""

    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Optional[Usage] = None


# ── Streaming response chunks ─────────────────────────────────────────────────


class DeltaToolCallFunction(BaseModel):
    """Partial function data inside a streaming tool-call delta."""

    name: Optional[str] = None
    # Incremental arguments fragment — concatenating all fragments gives valid JSON
    arguments: Optional[str] = None


class DeltaToolCall(BaseModel):
    """One entry in delta.tool_calls[] within a streaming chunk."""

    index: int
    # Only present in the *first* delta for this tool call
    id: Optional[str] = None
    type: Optional[str] = None
    function: Optional[DeltaToolCallFunction] = None


class Delta(BaseModel):
    """The incremental content of a streaming assistant turn."""

    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[DeltaToolCall]] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """A single SSE chunk in a streaming chat completion response."""

    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]
    # Only sent in the final chunk (finish_reason is not None)
    usage: Optional[Usage] = None
