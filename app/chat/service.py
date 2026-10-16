"""
One chat turn: search, then answer.

    message + history ─> search (query understanding + hybrid)
                           ├─ out of scope ─────────────> fixed reply (no LLM)
                           ├─ no results ───────────────> fixed reply with the filters used
                           └─> product details ─> draft (LLM) ─> validate ─> ChatAnswer
                                                     └ error / timeout ─> top results, fixed message

The API is stateless: the client sends the earlier turns with every request.
Query understanding sees the shopper's earlier messages (to resolve "cheaper
ones?"); the answer model sees the last few turns of both sides.
"""

from __future__ import annotations

import time

from psycopg_pool import AsyncConnectionPool

from app.core.config import get_settings
from app.generation.context import build_context, load_details
from app.generation.generate import (
    draft_answer,
    fallback_answer,
    no_results_answer,
    out_of_scope_answer,
    validate,
)
from app.generation.prompt import PROMPT_VERSION
from app.generation.schemas import ChatAnswer, Turn
from app.query.search import search
from app.schemas.query_models import SearchFilters

_TURN_CHARS = 600


def _describe_filters(qu: dict, explicit: SearchFilters) -> dict:
    """The filters actually applied, in words the model (and the shopper) can read."""
    applied = {**explicit.model_dump(mode="json", exclude_defaults=True), **(qu.get("inferred_filters") or {})}
    if qu.get("relaxed"):
        applied = explicit.model_dump(mode="json", exclude_defaults=True)
    applied.pop("in_stock_only", None)
    return {k: v for k, v in applied.items() if v not in (None, [], False)}


async def chat(
    pool: AsyncConnectionPool,
    message: str,
    history: list[Turn] | None = None,
    filters: SearchFilters | None = None,
) -> ChatAnswer:
    s = get_settings()
    started = time.perf_counter()
    history = list(history or [])[-s.chat_history_turns:]
    explicit = filters or SearchFilters()

    res = await search(pool, message, explicit, limit=s.generation_candidates,
                       history=[t.content for t in history if t.role == "user"])
    qu = res.debug.get("qu", {})
    timings = {"search": res.timings_ms.get("total", 0.0)}
    applied = _describe_filters(qu, explicit)

    def finish(answer: ChatAnswer, **debug) -> ChatAnswer:
        timings["total"] = round((time.perf_counter() - started) * 1000, 1)
        answer.timings_ms = {**timings, **answer.timings_ms}
        answer.debug = {"qu": qu, "applied_filters": applied,
                        "retrieved": [r.product.product_id for r in res.results], **debug}
        return answer

    if qu.get("out_of_scope"):
        return finish(out_of_scope_answer())
    cards = [r.product for r in res.results]
    if not cards:
        return finish(no_results_answer(applied))

    t0 = time.perf_counter()
    details = await load_details(pool, [c.product_id for c in cards])
    context = build_context(cards, details)
    timings["context"] = round((time.perf_counter() - t0) * 1000, 1)

    turns = [(t.role, t.content[:_TURN_CHARS]) for t in history]
    try:
        draft = await draft_answer(message, turns, context, applied)
    except Exception as e:     # noqa: BLE001 - any failure: show the results without an LLM answer
        return finish(fallback_answer(cards, warnings=[f"generation failed: {type(e).__name__}: {e}"[:300]]))

    timings["generate"] = round(draft.latency_ms, 1)
    answer = validate(draft.answer, cards, s.generation_max_picks)
    return finish(answer, prompt=PROMPT_VERSION, model=s.generation_model,
                  tokens={"input": draft.input_tokens, "output": draft.output_tokens},
                  draft=draft.answer.model_dump(mode="json"))
