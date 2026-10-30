"""
The FastAPI app.

    uvicorn app.api.main:create_app --factory --reload               # local
    uvicorn app.api.main:create_app --factory --host 0.0.0.0 --port 8000 \\
        --proxy-headers --forwarded-allow-ips="*" --no-access-log    # behind a proxy (Azure)

--factory: the app is built when the server starts, so settings are read then
(and importing this module in tests needs no environment). --no-access-log:
the request-context middleware writes its own access log, with request ids.
--proxy-headers: the rate limit keys on the real client IP, not the proxy's.

Startup (lifespan) opens the database pool and warms what the first request
would otherwise pay for: the catalog vocabulary and the OpenAI client. The API
refuses to start without a database rather than failing every request.
Interactive docs: /docs.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from psycopg_pool import PoolTimeout

from app.api.errors import install_error_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.ratelimit import SlidingWindowLimiter
from app.api.routes import chat, health, search
from app.core.config import get_settings
from app.core.db import create_pool
from app.core.llm import get_openai_client
from app.query.vocabulary import load_vocabulary

log = logging.getLogger("api")

API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    s = get_settings()
    pool = create_pool(min_size=s.api_db_pool_min_size, max_size=s.api_db_pool_max_size)
    try:
        await pool.open(wait=True, timeout=s.api_db_connect_timeout_seconds)
    except PoolTimeout:
        await pool.close()
        # Not the URL: it holds the password.
        log.error("could not connect to the database within %.0f s; check DATABASE_URL and that "
                  "Postgres is running (docker compose up -d)", s.api_db_connect_timeout_seconds)
        raise
    app.state.pool = pool
    try:
        vocab = await load_vocabulary(pool)
        get_openai_client()
        log.info("ready: pool %d-%d, vocabulary %d categories / %d brands, models %s / %s",
                 s.api_db_pool_min_size, s.api_db_pool_max_size, len(vocab.categories), len(vocab.brands),
                 s.query_model, s.generation_model)
        yield
    finally:
        await pool.close()
        log.info("shut down: pool closed")


def _configure_logging() -> None:
    """Our loggers print even when uvicorn's logging config leaves the root logger bare."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def create_app(*, use_lifespan: bool = True) -> FastAPI:
    """use_lifespan=False lets tests attach a fake pool to app.state instead of a database."""
    s = get_settings()
    _configure_logging()
    app = FastAPI(
        title="Kith Hybrid RAG API",
        version=API_VERSION,
        description="Hybrid (BM25 + vector) product search and a grounded shopping assistant over the Kith catalog.",
        lifespan=lifespan if use_lifespan else None,
    )
    app.state.chat_limiter = SlidingWindowLimiter(s.api_chat_rate_limit_per_minute)

    # Starlette runs the last-added middleware first: request context wraps everything,
    # so CORS preflights and errors also get a request id and an access-log line.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.api_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Retry-After", "Server-Timing"],
        max_age=600,
    )
    app.add_middleware(RequestContextMiddleware)
    install_error_handlers(app)

    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    return app
