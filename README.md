# OpenAI-Compatible Gateway

A FastAPI adapter that exposes an OpenAI Chat Completions-compatible API on top
of a configured upstream chat API.

This project is the translation layer between:

- OpenAI-compatible clients such as opencode
- The real upstream API contract your backend actually implements

Its job is to adapt request/response formats, streaming behavior, and tool-call
conventions without forcing the upstream service itself to speak OpenAI natively.

---

## Architecture

```
opencode / any OpenAI client
        │
        │  POST /v1/chat/completions  (OpenAI format)
        ▼
┌──────────────────────────────────┐
│        Adapter Gateway           │
│                                  │
│  1. Accept OpenAI request        │
│  2. Convert → upstream contract  │
│  3. Inject tool schema if needed │
│  4. Stream from upstream API     │
│  5. Adapt output → OpenAI        │
└──────────────────────────────────┘
        │
        │  POST /chat  {"query": "...", "history": "[...]"}
        ▼
┌──────────────────────────────────┐
│           Upstream API           │
└──────────────────────────────────┘
```

---

## Project Structure

```
openai-compatible-gateway/
├── main.py                  ← FastAPI app, utility routes, entry point
├── config.py                ← Environment-based upstream, retry, and gateway settings
├── pyproject.toml           ← Project metadata and dependencies
├── uv.lock                  ← Locked dependency set for uv
├── routers/
│   └── chat.py              ← POST /v1/chat/completions handler
├── services/
│   └── upstream.py          ← Async httpx client, SSE parser, retry handling
├── utils/
│   ├── converters.py        ← OpenAI ↔ upstream format conversion, tool-call detection
│   └── prompt_injector.py   ← Builds the tool-injection system prompt
├── schemas/
│   └── openai.py            ← Pydantic v2 OpenAI-compatible schemas
├── tests/                   ← Router, converter, upstream, prompt regression tests
├── .env.example
├── .python-version          ← Local development Python version
└── README.md
```

---

## Installation

**Python 3.12+ is required**. The repository's `.python-version` currently
uses Python 3.14 for local development.

This repository is defined by `pyproject.toml` and `uv.lock`; it does **not**
ship a `requirements.txt`.

```bash
# Install runtime + dev dependencies from the lockfile
uv sync --dev
```

---

## Configuration

```bash
cp .env.example .env
# Edit .env with your upstream API details
```

The example file lists the commonly edited values. The full set of settings
loaded by `config.py` is:

| Variable                    | Default                 | Description                                                 |
| --------------------------- | ----------------------- | ----------------------------------------------------------- |
| `UPSTREAM_BASE_URL`         | `http://localhost:8000` | Base URL of the upstream chat API, without a trailing slash |
| `UPSTREAM_API_KEY`          | _(empty)_               | Fallback bearer token for the upstream API                  |
| `UPSTREAM_CHAT_ENDPOINT`    | `/chat`                 | Path appended to `UPSTREAM_BASE_URL` for chat requests      |
| `UPSTREAM_TIMEOUT`          | `120.0`                 | Per-request upstream timeout in seconds                     |
| `UPSTREAM_RETRY_ATTEMPTS`   | `3`                     | Total attempts for upstream 5xx/network failures            |
| `UPSTREAM_RETRY_BASE_DELAY` | `0.5`                   | Base delay in seconds for exponential retry backoff         |
| `LOG_LEVEL`                 | `INFO`                  | Python logging level                                        |
| `GATEWAY_HOST`              | `0.0.0.0`               | Host used by the built-in `python main.py` runner           |
| `GATEWAY_PORT`              | `8080`                  | Port used by the built-in `python main.py` runner           |
| `MAX_REQUEST_BODY_SIZE`     | `10485760`              | Maximum HTTP request body size in bytes before 413          |

If an incoming client request includes `Authorization: Bearer <token>`, that
token is forwarded to the upstream API for that request. Otherwise the gateway
uses `UPSTREAM_API_KEY` when it is set.

---

## Running the Gateway

```bash
# Development (auto-reload on file changes)
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload

# Or via the built-in runner
uv run python main.py

# Production
uv run uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4
```

The gateway will be available at `http://localhost:8080`.
Interactive API docs: `http://localhost:8080/docs`

Additional utility endpoints:

- `GET /health`
- `GET /v1/models` (synthetic compatibility stub)
- `GET /` (gateway metadata and endpoint summary)

The FastAPI app allows all CORS origins for local tool compatibility. It also
rejects requests larger than `MAX_REQUEST_BODY_SIZE` with an OpenAI-style 413
error body before routing the request.

---

## Adapter Responsibilities

This gateway is responsible for:

1. Accepting OpenAI Chat Completions requests on `/v1/chat/completions`
2. Translating `messages` into the upstream API's `query` + `history` contract
3. Injecting tool definitions into the effective system prompt when `tools` are present
4. Sending only the upstream-supported `query` and `history` string fields
5. Reading upstream SSE payloads and normalising them into text and usage events
6. Detecting tool-call JSON and converting it into OpenAI `tool_calls`
7. Returning either OpenAI-style JSON or OpenAI streaming SSE back to the client
8. Retrying upstream 5xx and network failures before response streaming starts

The gateway is **not** the model backend itself. The upstream API remains the
source of truth for inference behavior, authentication requirements, and wire
contract details on the backend side.

---

## OpenAI Compatibility Scope

The current adapter actively uses these request fields:

- `model`
- `messages`
- `tools`
- `stream`

The request schema also accepts these common Chat Completions fields for client
compatibility:

- `n`: only `1` is supported; other values return HTTP 422
- `tool_choice`: accepted but not currently interpreted
- `user`: accepted but not currently forwarded
- `temperature`, `max_tokens`, `top_p`, `stop`, `presence_penalty`,
  `frequency_penalty`: accepted but not forwarded to the upstream API

Message `content` is modeled as plain text. Multimodal message parts and other
newer OpenAI Chat Completions options are outside the current schema.

---

## Configuring opencode

In your opencode configuration (`~/.config/opencode/config.json` or
`opencode.json` in the project root), add a custom provider pointing at
the gateway:

```json
{
  "providers": {
    "gateway-adapter": {
      "name": "Upstream Adapter Gateway",
      "baseURL": "http://localhost:8080",
      "models": [
        {
          "id": "gateway-adapter",
          "name": "Upstream Adapter",
          "contextLength": 200000
        }
      ]
    }
  }
}
```

> **Important:** `@ai-sdk/openai-compatible` automatically appends `/v1` to
> the `baseURL`, so set it to `http://localhost:8080` (without `/v1`).

---

## How Tool Calling Works

Since the upstream API has no native tool-calling surface, the gateway implements
tool calling via **prompt injection**:

1. When the OpenAI request contains `tools`, the gateway serialises the full
   tool schema into a system prompt and prepends it to the conversation history.

2. The system prompt instructs the upstream model to respond with a specific
   JSON format when it wants to call a tool:

   ```json
   {
     "tool_calls": [
       {
         "id": "call_abc123",
         "type": "function",
         "function": {
           "name": "tool_name",
           "arguments": "{\"param\": \"value\"}"
         }
       }
     ]
   }
   ```

3. The gateway detects this JSON in the upstream streaming output, parses it, and
   converts it to `delta.tool_calls` chunks in the OpenAI streaming format.

4. opencode receives the tool call, executes it, and sends the result back as
   a `role: "tool"` message. The gateway folds this into the history for the
   next upstream request.

---

## Verifying with curl

```bash
# Non-streaming
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gateway-adapter",
    "messages": [{"role": "user", "content": "Hello!"}]
  }' | jq .

# Streaming
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gateway-adapter",
    "stream": true,
    "messages": [{"role": "user", "content": "Count from 1 to 5."}]
  }'
```

---

## Upstream API Contract

The gateway currently sends the upstream API:

```http
POST /chat
{
  "query": "current user message",
  "history": "[{\"role\":\"system\",\"content\":\"...\"},{\"role\":\"user\",\"content\":\"...\"}]"
}
```

`query` and `history` are always strings. `history` is JSON-encoded with
`json.dumps(history, ensure_ascii=False)`, and an empty history is sent as the
string `"[]"`. The upstream request body contains no `stream` flag and no
OpenAI generation parameters.

The upstream request includes:

- `Accept: text/event-stream`
- `Content-Type: application/json`
- `Authorization: Bearer <token>` when either a client bearer token or
  `UPSTREAM_API_KEY` is available

The gateway expects the upstream API to respond with SSE-style events. Supported
text event payload formats:

| Format        | Example                                                                    |
| ------------- | -------------------------------------------------------------------------- |
| Claude native | `{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}` |
| Simple delta  | `{"delta":{"text":"Hi"}}`                                                  |
| Flat text     | `{"text":"Hi"}`                                                            |
| OpenAI-like   | `{"choices":[{"delta":{"content":"Hi"}}]}`                                 |
| Raw string    | `Hi`                                                                       |

Supported usage payload formats:

| Format                 | Extracted fields                                                        |
| ---------------------- | ----------------------------------------------------------------------- |
| Claude `message_start` | `input_tokens` → `prompt_tokens`, `output_tokens` → `completion_tokens` |
| Claude `message_delta` | `output_tokens` → `completion_tokens`                                   |
| OpenAI-like `usage`    | `prompt_tokens`, `completion_tokens`                                    |

When the upstream provides usage data, the gateway returns it as OpenAI-style
`usage` in non-streaming responses and in the final streaming chunk. Without
upstream usage data, `usage` is `null`.

Retry behavior is limited to failures that occur before any text has been
yielded to the client. HTTP 5xx responses and network errors are retried with
exponential backoff. HTTP 4xx responses are propagated immediately, and failures
after streaming has started are not retried to avoid duplicate output.

---

## Adjusting to the Real Upstream API Spec

If the real upstream API spec differs from the contract above, update the
gateway in the file that owns that concern instead of patching random spots.

| If the real upstream spec differs in...                                     | Edit here                                                                                                          | What to change                                                                                                                                              |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Base URL, endpoint path, API key, timeout, retry policy                     | `config.py`, `services/upstream.py`                                                                                | Update the settings and how the request URL, auth headers, timeout, and retry loop are built.                                                               |
| Request body field names such as `query` and `history`                      | `services/upstream.py`                                                                                             | Change the `payload` dict in `_stream_single_attempt()`.                                                                                                    |
| OpenAI generation parameters should affect upstream behavior                | `routers/chat.py`, `services/upstream.py`                                                                          | Add explicit forwarding only after the upstream API supports those fields.                                                                                  |
| How OpenAI messages should be mapped into the upstream request              | `utils/converters.py`                                                                                              | Update `messages_to_upstream_format()` so the last user message, system prompt, assistant turns, and tool results are converted to the real upstream shape. |
| History item schema, role names, or conversation ordering                   | `utils/converters.py`, `services/upstream.py`                                                                      | Change the `history` entries and whether they are sent as a JSON string or native array.                                                                    |
| Upstream streaming text or usage event shape                                | `services/upstream.py`                                                                                             | Update `_extract_text_from_sse_data()` and `_extract_usage_from_sse_data()` to read the real upstream event payloads.                                       |
| Upstream does not use SSE, or uses a different streaming transport          | `services/upstream.py`, `routers/chat.py`                                                                          | Change how `stream_upstream_response()` reads the upstream response and how `/v1/chat/completions` turns that into OpenAI-compatible streaming chunks.      |
| Tool calling must be instructed differently upstream                        | `utils/prompt_injector.py`                                                                                         | Rewrite the injected system prompt that explains how the upstream model should emit tool calls.                                                             |
| Tool-call output format differs from the current JSON object                | `utils/converters.py`, `routers/chat.py`                                                                           | Update `try_parse_tool_calls()`, `tool_calls_to_openai()`, `make_delta_tool_calls()`, and the final tool-call handling in the router.                       |
| Upstream returns full JSON responses instead of plain text / tool-call JSON | `routers/chat.py`, `services/upstream.py`                                                                          | Adjust the non-streaming and streaming assembly logic so the OpenAI response is built from the real upstream payload.                                       |
| OpenAI-facing request/response schema itself must change                    | `schemas/openai.py`, `routers/chat.py`                                                                             | Update the Pydantic models first, then update the router to emit the new fields consistently.                                                               |
| Gateway request limits or utility endpoints must change                     | `main.py`, `config.py`                                                                                             | Update middleware, utility routes, and any related settings.                                                                                                |
| Tests fail after the upstream spec change                                   | `tests/test_upstream.py`, `tests/test_converters.py`, `tests/test_chat_router.py`, `tests/test_prompt_injector.py` | Update or add regression tests that reflect the new upstream contract and the OpenAI-facing behavior.                                                       |

### Practical Change Order

When adapting to the actual upstream API, change the code in this order:

1. Update configuration names or endpoint details in `config.py`.
2. Update the upstream request payload and transport handling in `services/upstream.py`.
3. Add upstream forwarding for OpenAI generation parameters only if the upstream API supports them.
4. Update message/history conversion rules in `utils/converters.py`.
5. If tool calling is affected, update `utils/prompt_injector.py`, `utils/converters.py`, and `routers/chat.py` together.
6. Update the OpenAI-facing schemas in `schemas/openai.py` only if the gateway contract must also change.
7. Update tests so the new upstream spec is locked in.

### Fastest Way to Map a Spec Difference to Code

- "The upstream wants different request JSON" → `services/upstream.py`
- "The upstream wants different conversation/history format" → `utils/converters.py`
- "The upstream streams a different text or usage event format" → `services/upstream.py`
- "The upstream auth, timeout, or retry policy changed" → `config.py` + `services/upstream.py`
- "The upstream emits tool calls differently" → `utils/prompt_injector.py` + `utils/converters.py` + `routers/chat.py`
- "The client-facing OpenAI schema must change too" → `schemas/openai.py`
- "The gateway request-size limit or utility routes changed" → `main.py` + `config.py`
