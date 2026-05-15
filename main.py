"""
OpenAI-Compatible Gateway
=========================
FastAPI application that adapts OpenAI Chat Completions API requests into the
configured upstream API's (query, history) format, handles tool-calling via
prompt injection, and returns streaming or non-streaming OpenAI-format responses.

Usage
-----
    uvicorn main:app --host 0.0.0.0 --port 8080 --reload

Or via the helper in this file:
    python main.py
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from routers.chat import router as chat_router


# ── Body-size limiter ─────────────────────────────────────────────────────────


class LimitRequestBodyMiddleware:
    """Reject requests whose body exceeds *max_bytes* with HTTP 413.

    Two-stage check:
    1. Fast path — reject immediately when Content-Length header exceeds limit.
    2. Slow path — buffer the streamed body, reject if accumulated bytes exceed
       limit, otherwise replay the buffered body to the inner application so
       the one-shot ASGI ``receive`` callable is not exhausted.
    """

    def __init__(self, app: Callable, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Coroutine[Any, Any, dict[str, Any]]],
        send: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        await self._send_413(send)
                        return
                except ValueError:
                    pass
                break

        chunks: list[bytes] = []
        total = 0
        more_body = True

        while more_body:
            message = await receive()
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_bytes:
                await self._send_413(send)
                return
            chunks.append(chunk)
            more_body = message.get("more_body", False)

        full_body = b"".join(chunks)
        replayed = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": full_body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _send_413(
        send: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        body = json.dumps(
            {
                "error": {
                    "message": "Request body too large",
                    "type": "invalid_request_error",
                    "code": "request_too_large",
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})


# ── Logging ───────────────────────────────────────────────────────────────────
settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="OpenAI-Compatible Gateway",
    description=(
        "A gateway that exposes an OpenAI Chat Completions-compatible API "
        "backed by a configured upstream chat API. Supports streaming, "
        "tool calling via prompt injection, and full message history."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins so local tools (opencode, curl, etc.) can reach the gateway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Outermost layer: reject bodies that exceed the configured limit before any
# routing or further middleware runs. Added last so Starlette places it first
# in the execution chain.
app.add_middleware(LimitRequestBodyMiddleware, max_bytes=settings.MAX_REQUEST_BODY_SIZE)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(chat_router)


# ── Utility endpoints ─────────────────────────────────────────────────────────


@app.get("/health", tags=["utility"])
async def health():
    """Quick liveness check — useful for container health probes."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/v1/models", tags=["utility"])
async def list_models():
    """
    Stub model listing endpoint.
    Returns a synthetic model list so that clients expecting /v1/models
    (like some OpenAI SDK callers) don't get a 404.
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "gateway-adapter",
                "object": "model",
                "created": 1700000000,
                "owned_by": "gateway",
            }
        ],
    }


@app.get("/", tags=["utility"])
async def root():
    return {
        "name": "OpenAI-Compatible Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat_completions": "POST /v1/chat/completions",
            "models": "GET /v1/models",
            "health": "GET /health",
        },
    }


# ── Global error handler ──────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception for %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": type(exc).__name__,
                "code": "internal_error",
            }
        },
    )


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(
        "Starting gateway on %s:%d  (upstream → %s)",
        settings.GATEWAY_HOST,
        settings.GATEWAY_PORT,
        settings.UPSTREAM_BASE_URL,
    )
    uvicorn.run(
        "main:app",
        host=settings.GATEWAY_HOST,
        port=settings.GATEWAY_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )
