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
        │  POST /chat  {"query": "...", "history": [...]}
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
├── config.py                ← Environment-based settings (pydantic-settings)
├── pyproject.toml           ← Project metadata and dependencies
├── uv.lock                  ← Locked dependency set for uv
├── routers/
│   └── chat.py              ← POST /v1/chat/completions handler
├── services/
│   └── upstream.py          ← Async httpx client for the upstream API, SSE parser
├── utils/
│   ├── converters.py        ← OpenAI ↔ upstream format conversion, tool-call detection
│   └── prompt_injector.py   ← Builds the tool-injection system prompt
├── schemas/
│   └── openai.py            ← Pydantic v2 OpenAI-compatible schemas
├── tests/                   ← Router, converter, upstream parser regression tests
├── .env.example
└── README.md
```

---

## Installation

**Python 3.12+ is required** (3.14 recommended).

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

| Variable                 | Default                 | Description                                     |
| ------------------------ | ----------------------- | ----------------------------------------------- |
| `UPSTREAM_BASE_URL`      | `http://localhost:8000` | Base URL of the internal upstream API           |
| `UPSTREAM_API_KEY`       | _(empty)_               | Bearer token for the upstream API (if required) |
| `UPSTREAM_CHAT_ENDPOINT` | `/chat`                 | Path to the upstream chat endpoint              |
| `UPSTREAM_TIMEOUT`       | `120.0`                 | Request timeout in seconds                      |
| `LOG_LEVEL`              | `INFO`                  | Python logging level                            |
| `GATEWAY_HOST`           | `0.0.0.0`               | Host to bind                                    |
| `GATEWAY_PORT`           | `8080`                  | Port to bind                                    |

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

---

## Adapter Responsibilities

This gateway is responsible for:

1. Accepting OpenAI Chat Completions requests on `/v1/chat/completions`
2. Translating `messages` into the upstream API's `query` + `history` contract
3. Injecting tool definitions into the effective system prompt when `tools` are present
4. Reading upstream SSE payloads and normalising them into text chunks
5. Detecting tool-call JSON and converting it into OpenAI `tool_calls`
6. Returning either OpenAI-style JSON or OpenAI streaming SSE back to the client

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

The request schema also accepts several common Chat Completions fields for
client compatibility, but they are not currently translated into the upstream
request payload:

- `tool_choice`
- `temperature`
- `max_tokens`
- `top_p`
- `n`
- `stop`
- `presence_penalty`
- `frequency_penalty`
- `user`

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

The gateway currently expects the upstream API to accept:

```json
POST /chat
{
  "query": "current user message",
  "history": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "stream": true
}
```

And respond with SSE events. Supported event payload formats:

| Format        | Example                                                                    |
| ------------- | -------------------------------------------------------------------------- |
| Claude native | `{"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}` |
| Simple delta  | `{"delta":{"text":"Hi"}}`                                                  |
| Flat text     | `{"text":"Hi"}`                                                            |
| OpenAI-like   | `{"choices":[{"delta":{"content":"Hi"}}]}`                                 |
| Raw string    | `Hi`                                                                       |

---

## Adjusting to the Real Upstream API Spec

If the real upstream API spec differs from the contract above, update the
gateway in the file that owns that concern instead of patching random spots.

| If the real upstream spec differs in...                                     | Edit here                                                                         | What to change                                                                                                                                              |
| --------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Base URL, endpoint path, API key, timeout                                   | `config.py`, `services/upstream.py`                                               | Update the `UPSTREAM_*` settings and how the request URL / auth headers are built.                                                                          |
| Request body field names such as `query`, `history`, `stream`               | `services/upstream.py`                                                            | Change the `payload` dict in `stream_upstream_response()`.                                                                                                  |
| How OpenAI messages should be mapped into the upstream request              | `utils/converters.py`                                                             | Update `messages_to_upstream_format()` so the last user message, system prompt, assistant turns, and tool results are converted to the real upstream shape. |
| History item schema, role names, or conversation ordering                   | `utils/converters.py`                                                             | Change the `history` entries that are produced for upstream.                                                                                                |
| Upstream streaming format or SSE event JSON shape                           | `services/upstream.py`                                                            | Update `_extract_text_from_sse_data()` to read the real upstream event payloads.                                                                            |
| Upstream does not use SSE, or uses a different streaming transport          | `services/upstream.py`, `routers/chat.py`                                         | Change how `stream_upstream_response()` reads the upstream response and how `/v1/chat/completions` turns that into OpenAI-compatible streaming chunks.      |
| Tool calling must be instructed differently upstream                        | `utils/prompt_injector.py`                                                        | Rewrite the injected system prompt that explains how the upstream model should emit tool calls.                                                             |
| Tool-call output format differs from the current JSON object                | `utils/converters.py`, `routers/chat.py`                                          | Update `try_parse_tool_calls()`, `tool_calls_to_openai()`, `make_delta_tool_calls()`, and the final tool-call handling in the router.                       |
| Upstream returns full JSON responses instead of plain text / tool-call JSON | `routers/chat.py`, `services/upstream.py`                                         | Adjust the non-streaming and streaming assembly logic so the OpenAI response is built from the real upstream payload.                                       |
| OpenAI-facing request/response schema itself must change                    | `schemas/openai.py`, `routers/chat.py`                                            | Update the Pydantic models first, then update the router to emit the new fields consistently.                                                               |
| Tests fail after the upstream spec change                                   | `tests/test_upstream.py`, `tests/test_converters.py`, `tests/test_chat_router.py` | Update or add regression tests that reflect the new upstream contract and the OpenAI-facing behavior.                                                       |

### Practical Change Order

When adapting to the actual upstream API, change the code in this order:

1. Update configuration names or endpoint details in `config.py`.
2. Update the upstream request payload and transport handling in `services/upstream.py`.
3. Update message/history conversion rules in `utils/converters.py`.
4. If tool calling is affected, update `utils/prompt_injector.py`, `utils/converters.py`, and `routers/chat.py` together.
5. Update the OpenAI-facing schemas in `schemas/openai.py` only if the gateway contract must also change.
6. Update tests so the new upstream spec is locked in.

### Fastest Way to Map a Spec Difference to Code

- "The upstream wants different request JSON" → `services/upstream.py`
- "The upstream wants different conversation/history format" → `utils/converters.py`
- "The upstream streams a different event format" → `services/upstream.py`
- "The upstream emits tool calls differently" → `utils/prompt_injector.py` + `utils/converters.py` + `routers/chat.py`
- "The client-facing OpenAI schema must change too" → `schemas/openai.py`
