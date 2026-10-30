"""
POST /v1/chat: one turn of the shopping assistant.

Stateless: the client sends the earlier turns with every request. If the
model is slow or fails, the answer still comes back (200) with the top search
results and a fixed message; `warnings` says what happened.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from psycopg_pool import AsyncConnectionPool

from app.api.deps import chat_rate_limit, get_pool, require_debug_allowed, settings_dep
from app.api.errors import request_id_of
from app.api.schemas import ERROR_RESPONSES, ChatRequest, ChatResponse, ErrorResponse
from app.chat.service import chat
from app.core.config import Settings

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(chat_rate_limit)],
    responses={**ERROR_RESPONSES, 429: {"model": ErrorResponse, "description": "Rate limited; see Retry-After"}},
)
async def chat_turn(
    body: ChatRequest,
    request: Request,
    pool: AsyncConnectionPool = Depends(get_pool),
    settings: Settings = Depends(settings_dep),
) -> ChatResponse:
    require_debug_allowed(body.debug, settings)
    answer = await chat(pool, body.message, body.history, body.filters)
    data = answer.model_dump()
    data["debug"] = answer.debug if body.debug else None
    return ChatResponse(request_id=request_id_of(request), **data)
