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
from typing import Any, Awaitable, Callable

from psycopg_pool import AsyncConnectionPool

from app.core import tracing
from app.core.config import get_settings
from app.generation.context import build_context, load_details
from app.generation.generate import (
    draft_answer,
    stream_answer,
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
        # Relaxing drops inferred filters but keeps excluded brands (resolve.relaxed_filters).
        applied = explicit.model_dump(mode="json", exclude_defaults=True)
        excluded = (qu.get("inferred_filters") or {}).get("exclude_brands")
        if excluded:
            applied["exclude_brands"] = excluded
    applied.pop("in_stock_only", None)
    return {k: v for k, v in applied.items() if v not in (None, [], False)}


Emit = Callable[[str, Any], Awaitable[None]]


async def chat(
    pool: AsyncConnectionPool,
    message: str,
    history: list[Turn] | None = None,
    filters: SearchFilters | None = None,
    include_context: bool = False,
) -> ChatAnswer:
    """`include_context` adds the product blocks the model saw to debug (the eval's judge needs them)."""
    return await _turn(pool, message, history, filters, include_context, emit=None)


async def chat_stream(
    pool: AsyncConnectionPool,
    message: str,
    history: list[Turn] | None = None,
    filters: SearchFilters | None = None,
    *,
    emit: Emit,
) -> ChatAnswer:
    """
    The same turn as chat(), reporting progress through emit(event, data) on the way:
        ("status", "searching")   search started
        ("status", "writing")     the model started writing
        ("delta", text)           the next piece of the answer's message
    The returned answer is the authoritative one, validated like chat()'s. If generation
    fails after some text was sent, it is the fallback; the client replaces the streamed
    text with answer.message either way.
    """
    return await _turn(pool, message, history, filters, False, emit=emit)


async def _turn(pool: AsyncConnectionPool, message: str, history: list[Turn] | None,
                filters: SearchFilters | None, include_context: bool, emit: Emit | None) -> ChatAnswer:
    s = get_settings()
    started = time.perf_counter()
    history = list(history or [])[-s.chat_history_turns:]
    explicit = filters or SearchFilters()

    if emit:
        await emit("status", "searching")
    res = await search(pool, message, explicit, limit=s.generation_candidates,
                       history=[t.content for t in history if t.role == "user"])
    qu = res.debug.get("qu", {})
    timings = {"search": res.timings_ms.get("total", 0.0)}
    applied = _describe_filters(qu, explicit)

    def finish(answer: ChatAnswer, **debug) -> ChatAnswer:
        timings["total"] = round((time.perf_counter() - started) * 1000, 1)
        answer.timings_ms = {**timings, **answer.timings_ms}
        answer.debug = {"qu": qu, "applied_filters": applied, "search_timings": res.timings_ms,
                        "retrieved": [r.product.product_id for r in res.results], **debug}
        return answer

    if qu.get("out_of_scope"):
        return finish(out_of_scope_answer())
    cards = [r.product for r in res.results]
    if not cards:
        return finish(no_results_answer(applied))

    t0 = time.perf_counter()
    with tracing.observe("load-details", as_type="retriever", input={"ids": [c.product_id for c in cards]}) as obs:
        details = await load_details(pool, [c.product_id for c in cards])
        context = build_context(cards, details)
        obs.update(output={"products": len(details), "context_chars": len(context)})
    timings["context"] = round((time.perf_counter() - t0) * 1000, 1)

    turns = [(t.role, t.content[:_TURN_CHARS]) for t in history]
    generation_started = time.perf_counter()
    try:
        if emit:
            await emit("status", "writing")
            draft = await stream_answer(message, turns, context, applied, on_delta=lambda t: emit("delta", t))
        else:
            draft = await draft_answer(message, turns, context, applied)
    except Exception as e:     # noqa: BLE001 - any failure: show the results without an LLM answer
        return finish(fallback_answer(cards, warnings=[f"generation failed: {type(e).__name__}: {e}"[:300]]))

    timings["generate"] = round(draft.latency_ms, 1)
    if draft.first_token_ms is not None:      # from the start of the turn, what the shopper waited
        timings["first_token"] = round((generation_started - started) * 1000 + draft.first_token_ms, 1)
    with tracing.observe("validate-answer", as_type="guardrail",
                         input={"kind": draft.answer.kind, "picks": [p.product_id for p in draft.answer.picks]}) as obs:
        answer = validate(draft.answer, cards, s.generation_max_picks)
        obs.update(output={"kind": answer.kind, "picks": [p.product.product_id for p in answer.products],
                           "warnings": answer.warnings},
                   level="WARNING" if answer.warnings else None)
    extra = {"context": context} if include_context else {}
    return finish(answer, prompt=PROMPT_VERSION, model=s.generation_model,
                  tokens={"input": draft.input_tokens, "cached": draft.cached_tokens, "output": draft.output_tokens},
                  draft=draft.answer.model_dump(mode="json"), **extra)
