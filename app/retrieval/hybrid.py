"""
Hybrid search: BM25 and vector search over the same filtered candidates,
combined with Reciprocal Rank Fusion.

    SearchQuery ─┬─> BM25 (ParadeDB) ────────────────┐
                 └─> embed query ─> vector (HNSW) ───┴─> RRF ─> product cards

BM25 starts immediately; the vector search starts as soon as the query is
embedded. Each runs on its own pooled connection.

`mode` runs one method alone, which is what the evals compare against.

Try it from the project root:
    python -m app.retrieval.hybrid "samba espadrille"
    python -m app.retrieval.hybrid "warm jacket for winter" --max-price 300 --gender womens unisex
    python -m app.retrieval.hybrid "samba espadrille" --mode all
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, TypeVar

from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from app.core import llm
from app.core.config import get_settings
from app.retrieval.dense import search_dense
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.sparse import Hit, search_sparse
from app.schemas.product_models import (
    PRODUCT_CARD_COLUMNS,
    ProductCard,
    RetrievedProduct,
    SearchResult,
)
from app.schemas.query_models import SearchMode, SearchQuery

T = TypeVar("T")

_CARDS_SQL = sql.SQL("SELECT {cols} FROM clean.products WHERE product_id = ANY(%(ids)s)").format(
    cols=sql.SQL(", ").join(sql.Identifier(c) for c in PRODUCT_CARD_COLUMNS)
)


async def _timed(timings: dict[str, float], name: str, fn: Callable[[], Awaitable[T]]) -> T:
    start = time.perf_counter()
    try:
        return await fn()
    finally:
        timings[name] = round((time.perf_counter() - start) * 1000, 1)


async def _fetch_cards(pool: AsyncConnectionPool, ids: list[int]) -> dict[int, ProductCard]:
    if not ids:
        return {}
    async with pool.connection() as conn:
        cur = await conn.execute(_CARDS_SQL, {"ids": ids})
        rows = await cur.fetchall()
    return {r["product_id"]: ProductCard(**r) for r in rows}


async def hybrid_search(
    pool: AsyncConnectionPool,
    query: SearchQuery,
    mode: SearchMode = "hybrid",
) -> SearchResult:
    s = get_settings()
    k = max(s.retrieval_candidates, query.limit)
    timings: dict[str, float] = {}

    async def run_sparse() -> list[Hit]:
        async with pool.connection() as conn:
            return await search_sparse(conn, query.text, query.filters, k)

    async def run_dense() -> list[Hit]:
        vec = await _timed(timings, "embed_query", lambda: llm.embed_query(query.text))
        async with pool.connection() as conn:
            return await _timed(timings, "dense_search", lambda: search_dense(conn, vec, query.filters, k))

    started = time.perf_counter()
    sparse_hits: list[Hit] = []
    dense_hits: list[Hit] = []
    if mode == "hybrid":
        sparse_hits, dense_hits = await asyncio.gather(
            _timed(timings, "bm25_search", run_sparse), run_dense()
        )
    elif mode == "bm25":
        sparse_hits = await _timed(timings, "bm25_search", run_sparse)
    elif mode == "dense":
        dense_hits = await run_dense()
    else:
        raise ValueError(f"unknown mode {mode!r}")

    bm25_rank = {h.product_id: h.rank for h in sparse_hits}
    dense_rank = {h.product_id: h.rank for h in dense_hits}

    # Rank the candidates
    if mode == "hybrid":
        fused = reciprocal_rank_fusion(
            {"bm25": [h.product_id for h in sparse_hits], "dense": [h.product_id for h in dense_hits]},
            k=s.rrf_k,
            limit=query.limit,
        )
        ranked = [(f.product_id, f.score) for f in fused]
    else:
        hits = sparse_hits if mode == "bm25" else dense_hits
        ranked = [(h.product_id, h.score) for h in hits[: query.limit]]

    cards = await _timed(timings, "fetch_cards", lambda: _fetch_cards(pool, [pid for pid, _ in ranked]))
    timings["total"] = round((time.perf_counter() - started) * 1000, 1)

    results = [
        RetrievedProduct(
            product=cards[pid],
            score=score,
            bm25_rank=bm25_rank.get(pid),
            dense_rank=dense_rank.get(pid),
        )
        for pid, score in ranked
        if pid in cards          # a product deleted between search and fetch is simply dropped
    ]
    return SearchResult(mode=mode, results=results, timings_ms=timings)


# ===========================================================================
# CLI: try queries from the terminal
# ===========================================================================

def _print_result(res: SearchResult) -> None:
    print(f"\n=== {res.mode.upper()}  ({res.timings_ms.get('total', 0):.0f} ms: "
          + ", ".join(f"{k} {v:.0f}" for k, v in res.timings_ms.items() if k != "total") + ")")
    if not res.results:
        print("  (no results)")
        return
    print(f"  {'#':>2}  {'BM25':>4}  {'DENSE':>5}  {'SCORE':>8}  {'PRICE':>8}  TITLE")
    for i, r in enumerate(res.results, start=1):
        p = r.product
        price = f"${p.price_min:,.0f}" + ("*" if p.on_sale else "")
        stock = "" if p.in_stock else "  [sold out]"
        bm = str(r.bm25_rank) if r.bm25_rank else "-"
        dn = str(r.dense_rank) if r.dense_rank else "-"
        print(f"  {i:>2}  {bm:>4}  {dn:>5}  {r.score:>8.4f}  {price:>8}  {p.title[:70]}{stock}")


async def _cli() -> None:
    import argparse

    from app.core.db import create_pool
    from app.schemas.query_models import SearchFilters

    ap = argparse.ArgumentParser(description="Search the catalog.")
    ap.add_argument("text", help="what to search for")
    ap.add_argument("--mode", choices=["hybrid", "bm25", "dense", "all"], default="hybrid")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--min-price", type=float)
    ap.add_argument("--max-price", type=float)
    ap.add_argument("--gender", nargs="+", default=[], choices=["mens", "womens", "unisex", "kids"])
    ap.add_argument("--category", nargs="+", default=[])
    ap.add_argument("--subcategory", nargs="+", default=[])
    ap.add_argument("--brand", nargs="+", default=[])
    ap.add_argument("--size", nargs="+", default=[])
    ap.add_argument("--on-sale", action="store_true")
    ap.add_argument("--include-sold-out", action="store_true")
    args = ap.parse_args()

    query = SearchQuery(
        text=args.text,
        limit=args.limit,
        filters=SearchFilters(
            min_price=args.min_price, max_price=args.max_price, genders=args.gender,
            categories=args.category, subcategories=args.subcategory, brands=args.brand,
            sizes=args.size, on_sale_only=args.on_sale, in_stock_only=not args.include_sold_out,
        ),
    )
    modes: list[SearchMode] = ["bm25", "dense", "hybrid"] if args.mode == "all" else [args.mode]

    async with create_pool(max_size=4) as pool:
        for mode in modes:
            _print_result(await hybrid_search(pool, query, mode))
    print("\n  BM25 / DENSE = rank each method gave the product ('-' = not in its top "
          f"{max(get_settings().retrieval_candidates, args.limit)}).  * = on sale.")


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":     # psycopg's async mode needs the selector loop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_cli())