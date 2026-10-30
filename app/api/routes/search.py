"""
POST /v1/search: product search without a written answer.

    understand on (default in hybrid mode): query understanding, then hybrid search
        (spelling fixes, inferred filters, intent weights; falls back to the raw
        text if the LLM is slow or fails)
    understand off, or mode bm25 / dense: the retrieval methods directly, no LLM
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from psycopg_pool import AsyncConnectionPool

from app.api.deps import get_pool, require_debug_allowed, settings_dep
from app.api.errors import request_id_of
from app.api.schemas import ERROR_RESPONSES, SearchRequest, SearchResponse
from app.core.config import Settings
from app.query.search import search as understand_and_search
from app.retrieval.hybrid import hybrid_search
from app.schemas.query_models import SearchQuery

router = APIRouter(prefix="/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse, responses=ERROR_RESPONSES)
async def search_products(
    body: SearchRequest,
    request: Request,
    pool: AsyncConnectionPool = Depends(get_pool),
    settings: Settings = Depends(settings_dep),
) -> SearchResponse:
    require_debug_allowed(body.debug, settings)
    if body.understand:
        result = await understand_and_search(pool, body.query, body.filters, limit=body.limit)
    else:
        result = await hybrid_search(pool, SearchQuery(text=body.query, filters=body.filters, limit=body.limit),
                                     body.mode)
    return SearchResponse(
        request_id=request_id_of(request),
        mode=result.mode,
        results=result.results,
        timings_ms=result.timings_ms,
        debug=result.debug if body.debug else None,
    )
