"""
GET /health        the process is up (for the container's liveness probe; no dependencies)
GET /health/ready  it can serve: database reachable, catalog vocabulary loaded (readiness probe)
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from psycopg_pool import AsyncConnectionPool

from app.api.deps import get_pool
from app.api.schemas import HealthResponse
from app.query.vocabulary import load_vocabulary

router = APIRouter(tags=["health"])

READY_TIMEOUT_SECONDS = 3.0


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
async def ready(pool: AsyncConnectionPool = Depends(get_pool)):
    checks: dict[str, str] = {}
    try:
        async def ping() -> None:
            async with pool.connection() as conn:
                await conn.execute("SELECT 1")
        await asyncio.wait_for(ping(), READY_TIMEOUT_SECONDS)
        checks["database"] = "ok"
    except Exception as e:  # noqa: BLE001 - any failure means not ready
        checks["database"] = f"error: {type(e).__name__}"
    if checks["database"] == "ok":
        try:
            vocab = await asyncio.wait_for(load_vocabulary(pool), READY_TIMEOUT_SECONDS)
            checks["vocabulary"] = f"ok ({len(vocab.categories)} categories, {len(vocab.brands)} brands)"
        except Exception as e:  # noqa: BLE001
            checks["vocabulary"] = f"error: {type(e).__name__}"
    ok = all(v.startswith("ok") for v in checks.values())
    body = HealthResponse(status="ok" if ok else "unavailable", checks=checks)
    return body if ok else JSONResponse(body.model_dump(), status_code=503)
