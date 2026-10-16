"""
The search entry point the API will call: understand the message, then run
hybrid search with the resulting plan.

    message ─> [style code?] ──────────────────────────────> hybrid search (BM25 route)
            └> understand (LLM) ─> resolve ─> hybrid search ─> too few results? ─> retry
                    │ timeout / error                               without inferred filters
                    └──────────────> hybrid search on the raw message (Round 1 rules)

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
        "tokens": {"input": u.input_tokens, "output": u.output_tokens},
    }

    if u.plan.out_of_scope:
        # Not a product request (or an attempt to steer the assistant): search the raw
        # message with the shopper's own filters only, and let generation decide what to say.
        res = await hybrid_search(pool, raw)
        res.timings_ms["understand"] = round(u.latency_ms, 1)
        return finish(res, {**qu, "out_of_scope": True})

    r = resolve(u.plan, explicit, vocab, raw.text)
    qu.update(search_text=r.text, inferred_filters=r.inferred, dropped=r.dropped, weights=r.weights)

    dense_text = raw.text if s.qu_dense_text == "original" else None
    qu["dense_text"] = s.qu_dense_text
    query = SearchQuery(text=r.text, filters=r.filters, limit=limit)
    res = await hybrid_search(pool, query, weights=r.weights, dense_text=dense_text)

    # A wrong inferred filter can hide everything good; never let it empty the page.
    if r.has_inferred_filters and len(res.results) < min(s.qu_relax_min_results, limit):
        relaxed = await hybrid_search(pool, SearchQuery(text=r.text, filters=explicit, limit=limit),
                                      weights=r.weights, dense_text=dense_text)
        if len(relaxed.results) > len(res.results):
            qu["relaxed"] = {"reason": f"{len(res.results)} results with inferred filters"}
            res = relaxed

    res.timings_ms["understand"] = round(u.latency_ms, 1)
    return finish(res, qu)


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