"""
The search entry point the API will call: understand the message, then run
hybrid search with the resulting plan.

    message ─> [style code?] ──────────────────────────────> hybrid search (BM25 route)
            └> understand (LLM) ─> resolve ─> hybrid search ─> too few results? ─> retry
            │       │ timeout / error                               without inferred filters
            │       └──────────────> hybrid search on the raw message (Round 1 rules)
            └> embed the message (single-turn), at the same time, ready for vector search

Everything that happened is recorded in result.debug["qu"], so a bad answer
can be traced to the plan, the filters or the retrieval.

Try it from the project root:
    python -m app.query.search "looking for a cozy hoodie for my girlfriend"
    python -m app.query.search "nupste puffer"
"""

from __future__ import annotations

import asyncio
import time

from psycopg_pool import AsyncConnectionPool

from app.core import llm, tracing
from app.core.config import get_settings
from app.query.resolve import resolve
from app.query.understanding import PROMPT_VERSION, understand
from app.query.vocabulary import load_vocabulary
from app.retrieval.hybrid import hybrid_search
from app.retrieval.query_analysis import is_code_query
from app.schemas.product_models import SearchResult
from app.schemas.query_models import SearchFilters, SearchQuery


async def search(
    pool: AsyncConnectionPool,
    message: str,
    filters: SearchFilters | None = None,
    limit: int = 10,
    history: list[str] | None = None,
) -> SearchResult:
    """One traced search step: understanding, resolving and retrieval nest under it."""
    explicit = filters or SearchFilters()
    trace_input = {"message": tracing.mask_pii(message), "history_turns": len(history or []), "limit": limit,
                   "filters": explicit.model_dump(mode="json", exclude_defaults=True)}
    with tracing.observe("search-products", as_type="chain", input=trace_input) as obs:
        res = await _search(pool, message, filters, limit, history)
        qu = res.debug.get("qu", {})
        obs.update(output={"results": [{"id": r.product.product_id, "title": r.product.title}
                                       for r in res.results[:10]]},
                   metadata={"qu_used": qu.get("used"), "qu_reason": qu.get("reason"),
                             "out_of_scope": qu.get("out_of_scope", False), "relaxed": "relaxed" in qu,
                             "inferred_filters": qu.get("inferred_filters"), "timings_ms": res.timings_ms})
        return res


async def _search(
    pool: AsyncConnectionPool,
    message: str,
    filters: SearchFilters | None = None,
    limit: int = 10,
    history: list[str] | None = None,
) -> SearchResult:
    s = get_settings()
    explicit = filters or SearchFilters()
    raw = SearchQuery(text=message, filters=explicit, limit=limit)
    started = time.perf_counter()

    def finish(res: SearchResult, qu: dict) -> SearchResult:
        res.timings_ms["total"] = round((time.perf_counter() - started) * 1000, 1)
        res.debug["qu"] = qu
        return res

    # Style codes: the LLM can't improve on an exact BM25 match.
    if not s.query_understanding or (s.route_codes_to_bm25 and is_code_query(raw.text)):
        reason = "disabled" if not s.query_understanding else "style code"
        return finish(await hybrid_search(pool, raw), {"used": False, "reason": reason})

    # Vector search will embed the shopper's own words (unless it's a follow-up, which
    # embeds the plan's standalone query), and that doesn't depend on the plan: start
    # it now so it runs while the LLM reads the message, instead of after (~0.5 s saved).
    # It also serves the fallback paths below, which search the raw message.
    if s.prefetch_query_embedding and s.qu_dense_text == "original" and not history:
        llm.start_embed_query(raw.text)

    try:
        vocab = await load_vocabulary(pool)
        u = await understand(raw.text, vocab, history)
    except (asyncio.TimeoutError, Exception) as e:     # noqa: BLE001 - any failure means "search without it"
        res = await hybrid_search(pool, raw)
        res.timings_ms["understand"] = round((time.perf_counter() - started) * 1000, 1)
        return finish(res, {"used": False, "reason": "error", "error": f"{type(e).__name__}: {e}"[:300]})

    qu: dict = {
        "used": True,
        "prompt": PROMPT_VERSION,
        "plan": u.plan.model_dump(mode="json"),
        "cached": u.cached,
        "tokens": {"input": u.input_tokens, "cached": u.cached_tokens, "output": u.output_tokens},
    }

    if u.plan.out_of_scope:
        # Not a product request (or an attempt to steer the assistant): search the raw
        # message with the shopper's own filters only, and let generation decide what to say.
        res = await hybrid_search(pool, raw)
        res.timings_ms["understand"] = round(u.latency_ms, 1)
        return finish(res, {**qu, "out_of_scope": True})

    with tracing.observe("resolve-filters", input={"plan_filters": {
            k: v for k, v in u.plan.model_dump(mode="json").items()
            if k in ("categories", "subcategories", "genders", "brands", "exclude_brands", "min_price",
                     "max_price", "on_sale_only", "audience", "intent")}}) as obs:
        r = resolve(u.plan, explicit, vocab, raw.text)
        obs.update(output={"search_text": r.text, "inferred_filters": r.inferred, "dropped": r.dropped,
                           "weights": r.weights})
    qu.update(search_text=r.text, inferred_filters=r.inferred, dropped=r.dropped, weights=r.weights)

    # Vector search reads natural language best, so it embeds the shopper's own words
    # rather than the keyword rewrite. A follow-up ("something cheaper") means nothing
    # alone, so then it embeds the plan's self-contained version ("a cheaper black hoodie").
    if s.qu_dense_text == "rewrite":
        dense_text = None
    elif history and u.plan.standalone_query.strip():
        dense_text = " ".join(u.plan.standalone_query.split())
    else:
        dense_text = raw.text
    qu["dense_text"] = dense_text or r.text

    items = [" ".join(i.split()) for i in u.plan.compare_items if i and i.strip()][:4]
    if len(items) >= 2:
        # "Samba OG vs Gazelle": one search finds the stronger match and buries the other.
        # Search each product on its own (brand filters dropped: the items may differ) and
        # interleave, so every compared product is in the candidates.
        item_filters = r.filters.model_copy(update={"brands": explicit.brands})
        per_item = max(2, limit // len(items))
        runs = await asyncio.gather(*(
            hybrid_search(pool, SearchQuery(text=it, filters=item_filters, limit=per_item))
            for it in items
        ))
        res = _interleave(runs, limit)
        qu["compare_items"] = items
        res.timings_ms["understand"] = round(u.latency_ms, 1)
        return finish(res, qu)
    query = SearchQuery(text=r.text, filters=r.filters, limit=limit)
    res = await hybrid_search(pool, query, weights=r.weights, dense_text=dense_text)

    # A wrong inferred filter can hide everything good; never let it empty the page.
    if r.has_inferred_filters and len(res.results) < min(s.qu_relax_min_results, limit):
        with tracing.observe("relax-filters", metadata={"results_with_inferred_filters": len(res.results)}):
            relaxed = await hybrid_search(pool, SearchQuery(text=r.text, filters=r.relaxed_filters, limit=limit),
                                          weights=r.weights, dense_text=dense_text)
        if len(relaxed.results) > len(res.results):
            qu["relaxed"] = {"reason": f"{len(res.results)} results with inferred filters"}
            res = relaxed

    res.timings_ms["understand"] = round(u.latency_ms, 1)
    return finish(res, qu)


def _interleave(runs: list[SearchResult], limit: int) -> SearchResult:
    """Round-robin over several result lists: 1st of each, then 2nd of each, ...; no duplicates."""
    seen: set[int] = set()
    merged = []
    for rank in range(max((len(r.results) for r in runs), default=0)):
        for r in runs:
            if rank < len(r.results) and r.results[rank].product.product_id not in seen:
                seen.add(r.results[rank].product.product_id)
                merged.append(r.results[rank])
    timings = {k: max(r.timings_ms.get(k, 0.0) for r in runs) for r in runs for k in r.timings_ms}
    return SearchResult(mode="hybrid", results=merged[:limit], timings_ms=timings,
                        debug={"route": "compare-items", "items": [r.debug for r in runs]})


# ===========================================================================
# CLI
# ===========================================================================

async def _cli() -> None:
    import argparse
    import json

    from app.core.db import create_pool
    from app.retrieval.hybrid import _print_result, add_filter_arguments, filters_from_args

    ap = argparse.ArgumentParser(description="Search with query understanding.")
    ap.add_argument("text")
    ap.add_argument("--limit", type=int, default=10)
    add_filter_arguments(ap)
    args = ap.parse_args()

    async with create_pool(max_size=4) as pool:
        res = await search(pool, args.text, filters_from_args(args), args.limit)
    qu = res.debug.pop("qu", {})
    print(json.dumps(qu, indent=2, default=str))
    _print_result(res)


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_cli())
