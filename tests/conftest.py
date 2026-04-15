"""
Shared pytest fixtures.

The async_client fixture spins up the FastAPI app in-process using
httpx.ASGITransport — no real network socket is needed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from main import app


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client wired directly to the FastAPI ASGI app."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ── SSE helpers ───────────────────────────────────────────────────────────────


async def iter_sse_chunks(response, /) -> AsyncIterator[dict]:
    """
    Async-iterate over a streaming response and yield parsed SSE data objects.
    Stops at [DONE].
    """
    async for raw_line in response.aiter_lines():
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        yield json.loads(payload)
