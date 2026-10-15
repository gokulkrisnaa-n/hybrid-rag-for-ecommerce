"""
Async Postgres connection pool for the live API.

The ingestion scripts use one plain connection each (ingestion/db.py); the API
serves many requests at once, so it borrows connections from a pool instead.
The FastAPI app opens the pool on startup and closes it on shutdown.
"""

from __future__ import annotations

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import get_settings


def create_pool(url: str | None = None, *, min_size: int = 1, max_size: int = 10) -> AsyncConnectionPool:
    """
    Create a pool (not yet opened). Open it with `await pool.open()` inside the
    running event loop, or use it as `async with create_pool() as pool:`.

    autocommit=True: single statements commit on their own, and
    `async with conn.transaction():` blocks are real transactions.
    """
    return AsyncConnectionPool(
        conninfo=url or get_settings().database_url,
        min_size=min_size,
        max_size=max_size,
        open=False,
        kwargs={"row_factory": dict_row, "autocommit": True},
    )