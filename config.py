"""
Gateway configuration — loaded from environment variables (or a .env file).

All settings related to the upstream API and gateway behaviour live here
so that no magic strings are scattered across the codebase.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── upstream chat API ─────────────────────────────────────────────────────
    # Base URL of the upstream chat API (no trailing slash)
    UPSTREAM_BASE_URL: str = Field(
        default="http://localhost:8000", description="Base URL for upstream API"
    )
    # Bearer token sent as Authorization header; leave blank if upstream needs no auth
    UPSTREAM_API_KEY: str = Field(
        default="", description="API key / bearer token for upstream"
    )
    # Path to the chat endpoint — appended to UPSTREAM_BASE_URL for each request
    UPSTREAM_CHAT_ENDPOINT: str = Field(
        default="/chat", description="Chat endpoint path on upstream"
    )
    # Per-request HTTP timeout in seconds (can take a while to start streaming)
    UPSTREAM_TIMEOUT: float = Field(
        default=120.0, description="Timeout for upstream API requests in seconds"
    )
    # Total number of attempts (1 = no retry).  Only 5xx and network errors are retried.
    UPSTREAM_RETRY_ATTEMPTS: int = Field(
        default=3,
        description="Total upstream attempts before giving up (1 disables retries)",
    )
    # Base delay in seconds for exponential backoff: delay = base * 2^attempt
    UPSTREAM_RETRY_BASE_DELAY: float = Field(
        default=0.5, description="Base delay (seconds) for exponential-backoff retries"
    )

    # ── gateway ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO", description="Python logging level")
    GATEWAY_HOST: str = Field(
        default="0.0.0.0", description="Host to bind the gateway server"
    )
    GATEWAY_PORT: int = Field(
        default=8080, description="Port to bind the gateway server"
    )
    # Maximum allowed request body size in bytes (default 10 MiB).
    # Requests with a larger Content-Length or actual body are rejected with 413.
    MAX_REQUEST_BODY_SIZE: int = Field(
        default=10 * 1024 * 1024,
        description="Maximum request body size in bytes (default 10 MiB)",
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    """Return a cached, singleton Settings instance (parsed once at first access)."""
    return Settings()
