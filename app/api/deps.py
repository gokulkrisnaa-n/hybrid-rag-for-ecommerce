"""Dependencies shared by the routes: the database pool, debug access, the chat rate limit."""

from __future__ import annotations

from fastapi import Request
from psycopg_pool import AsyncConnectionPool

from app.api.errors import ApiError
from app.core.config import Settings, get_settings


def get_pool(request: Request) -> AsyncConnectionPool:
    """The pool the app opened at startup (main.lifespan)."""
    return request.app.state.pool


def settings_dep() -> Settings:
    return get_settings()


def require_debug_allowed(debug: bool, settings: Settings) -> None:
    """debug output shows how the system works inside; only where the deployment allows it."""
    if debug and not settings.api_debug_enabled:
        raise ApiError(403, "debug_disabled", "Debug output is turned off on this server.")


async def chat_rate_limit(request: Request) -> None:
    """Per client IP. Behind a proxy, run uvicorn with --proxy-headers so this is the real client."""
    client = request.client.host if request.client else "unknown"
    allowed, retry_after = request.app.state.chat_limiter.check(client)
    if not allowed:
        raise ApiError(429, "rate_limited",
                       f"Too many chat requests. Please wait {retry_after} s and try again.",
                       headers={"Retry-After": str(retry_after)})
