"""
POST /v1/chat: one turn of the shopping assistant.

Stateless: the client sends the earlier turns with every request. If the
model is slow or fails, the answer still comes back (200) with the top search
results and a fixed message; `warnings` says what happened.

Each request is one trace (id = request_id) with three automatic scores, so
bad answers can be filtered in Langfuse without reading every trace:
    answer-fallback   BOOLEAN      1 if the shopper got the fixed "closest matches" reply
    answer-warnings   NUMERIC      problems validation fixed (dropped picks, prices in text, ...)
    answer-kind       CATEGORICAL  recommend / compare / no_match / clarify / out_of_scope
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from psycopg_pool import AsyncConnectionPool

from app.api.deps import chat_rate_limit, get_pool, require_debug_allowed, settings_dep
from app.api.errors import request_id_of
from app.api.schemas import ERROR_RESPONSES, ChatRequest, ChatResponse, ErrorResponse
from app.chat.service import chat
from app.core import tracing
from app.core.config import Settings
from app.generation.prompt import PROMPT_VERSION as GEN_PROMPT
from app.generation.schemas import ChatAnswer
from app.query.understanding import PROMPT_VERSION as QU_PROMPT

router = APIRouter(prefix="/v1", tags=["chat"])


def is_fallback(answer: ChatAnswer) -> bool:
    """Generation failed or gave nothing usable, and the shopper got the fixed reply.
    A deliberate fixed reply (out of scope, no results) is not a fallback."""
    return answer.kind == "error" or any(w.startswith(("generation failed", "no valid picks"))
                                         for w in answer.warnings)


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
    request_id = request_id_of(request)
    trace_input = {
        "message": tracing.mask_pii(body.message),
        "history": [{"role": t.role, "content": tracing.mask_pii(t.content)} for t in body.history],
        "filters": body.filters.model_dump(mode="json", exclude_defaults=True),
    }
    with tracing.request_trace("chat", request_id=request_id, input=trace_input, session_id=body.session_id,
                               tags=["chat", f"qu:{QU_PROMPT}", f"gen:{GEN_PROMPT}"]) as root:
        answer = await chat(pool, body.message, body.history, body.filters)
        root.update(output={"kind": answer.kind, "message": answer.message,
                            "products": [{"id": p.product.product_id, "title": p.product.title}
                                         for p in answer.products],
                            "warnings": answer.warnings},
                    metadata={"timings_ms": answer.timings_ms})
        root.score("answer-fallback", 1 if is_fallback(answer) else 0, data_type="BOOLEAN")
        root.score("answer-warnings", len(answer.warnings), data_type="NUMERIC")
        root.score("answer-kind", answer.kind, data_type="CATEGORICAL")

    data = answer.model_dump()
    data["debug"] = answer.debug if body.debug else None
    return ChatResponse(request_id=request_id, **data)
