"""
Label evaluation queries.

Each query runs through BM25, vector and hybrid search; their top results are
pooled into one numbered list (the standard "pooling" method for building
search test sets) and you say which ones are relevant. Cases are saved to
evals/ground_truth.jsonl.

Work through the planned queries (resumes where you left off):
    python -m evals.label --plan

Or label a single query:
    python -m evals.label "samba espadrille" --tag exact-name
    python -m evals.label "warm jacket for winter" --gender womens unisex --max-price 300 --tag vague-intent

At the prompt (numbers refer to the list):
    1 2 5          products 1, 2 and 5 are exactly right (grade 2)
    7=1 9=1        7 and 9 are acceptable alternatives (grade 1)
    1-4  /  1-4=1  ranges
    8286850580608  a product ID (6+ digits) that none of the methods found; add =1 for grade 1
    s              skip this query (plan mode; it comes back next time)
    q              quit (plan mode; everything labeled so far is already saved)
    (blank)        skip in plan mode, cancel for a single query
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys

from psycopg_pool import AsyncConnectionPool

from app.core.db import create_pool
from app.retrieval.hybrid import add_filter_arguments, filters_from_args, hybrid_search
from app.schemas.product_models import ProductCard
from app.schemas.query_models import SearchQuery
from evals.dataset import (
    DEFAULT_PATH, PLAN_PATH, EvalCase, PlannedQuery, load_cases, load_plan, save_cases, slugify,
)

MODES = ("bm25", "dense", "hybrid")
_TOKEN = re.compile(r"^(\d+)(?:-(\d+))?(?:=([12]))?$")
PRODUCT_ID_MIN_DIGITS = 6        # list positions are small; Shopify product IDs are long

Candidates = list[tuple[ProductCard, dict[str, int]]]


# ===========================================================================
# Parsing answers
# ===========================================================================

def parse_answer(text: str, n_items: int) -> tuple[dict[int, int], dict[int, int]]:
    """
    Returns (list positions -> grade, extra product IDs -> grade).
    '1 2 5=1 7-9 8286850580608=1' -> ({1: 2, 2: 2, 5: 1, 7: 2, 8: 2, 9: 2}, {8286850580608: 1})
    Raises ValueError on anything it can't read.
    """
    positions: dict[int, int] = {}
    extra: dict[int, int] = {}
    for tok in re.split(r"[\s,]+", text.strip()):
        if not tok:
            continue
        m = _TOKEN.match(tok)
        if not m:
            raise ValueError(f"can't read {tok!r}; use forms like 3, 3=1, 2-5, 2-5=1 or a product ID")
        lo_s, hi_s, g = m.group(1), m.group(2), int(m.group(3) or 2)
        if hi_s is None and len(lo_s) >= PRODUCT_ID_MIN_DIGITS:
            extra[int(lo_s)] = g
            continue
        lo, hi = int(lo_s), int(hi_s or lo_s)
        if lo > hi:
            raise ValueError(f"range {tok!r} is backwards")
        for i in range(lo, hi + 1):
            if not 1 <= i <= n_items:
                raise ValueError(f"{i} is not in the list (1-{n_items})")
            positions[i] = g
    return positions, extra


def parse_grades(text: str, n_items: int) -> dict[int, int]:
    """List positions only (product IDs are rejected). Kept for callers that don't expect IDs."""
    positions, extra = parse_answer(text, n_items)
    if extra:
        raise ValueError(f"{next(iter(extra))} is not in the list (1-{n_items})")
    return positions


def parse_also(values: list[str]) -> dict[int, int]:
    out = {}
    for v in values:
        pid, _, g = v.partition("=")
        if not pid.isdigit() or g not in ("", "1", "2"):
            raise ValueError(f"--also expects PRODUCT_ID or PRODUCT_ID=GRADE, got {v!r}")
        out[int(pid)] = int(g or 2)
    return out


# ===========================================================================
# Pooling and display
# ===========================================================================

async def pool_candidates(pool: AsyncConnectionPool, query: SearchQuery) -> Candidates:
    """Union of every mode's top results, best-ranked first, with each mode's rank."""
    cards: dict[int, ProductCard] = {}
    ranks: dict[int, dict[str, int]] = {}
    for mode in MODES:
        res = await hybrid_search(pool, query, mode)
        for rank, r in enumerate(res.results, start=1):
            cards[r.product.product_id] = r.product
            ranks.setdefault(r.product.product_id, {})[mode] = rank
    order = sorted(cards, key=lambda pid: (min(ranks[pid].values()), ranks[pid].get("hybrid", 999), pid))
    return [(cards[pid], ranks[pid]) for pid in order]


def print_candidates(items: Candidates) -> None:
    if not items:
        print("  No results in any mode.")
        return
    print(f"\n  {'#':>3}  {'BM25':>4} {'DENSE':>5} {'HYBR':>4}  {'PRICE':>7}  {'GENDER':<7} {'CATEGORY':<30} TITLE")
    for n, (p, r) in enumerate(items, start=1):
        cols = [str(r[m]) if m in r else "-" for m in MODES]
        cat = "/".join(x for x in (p.category, p.subcategory) if x)[:30]
        price = f"${p.price_min:,.0f}"
        stock = "" if p.in_stock else "  [sold out]"
        print(f"  {n:>3}  {cols[0]:>4} {cols[1]:>5} {cols[2]:>4}  {price:>7}  "
              f"{(p.gender or '-'):<7} {cat:<30} {p.title[:70]}{stock}")


def print_query_header(query: SearchQuery) -> None:
    print(f"\nQuery: {query.text!r}")
    print(f"Filters: {query.filters.model_dump(mode='json', exclude_defaults=True) or 'none (in stock only)'}")


def build_case(case_id: str, query: SearchQuery, items: Candidates, positions: dict[int, int],
               extra: dict[int, int], tags: list[str], notes: str) -> EvalCase | None:
    relevant = {items[n - 1][0].product_id: g for n, g in positions.items()}
    relevant.update(extra)
    if not relevant:
        return None
    return EvalCase(id=case_id, query=query.text, filters=query.filters,
                    relevant=relevant, tags=tags, notes=notes)


def upsert_case(case: EvalCase) -> int:
    """Insert or replace by id and save immediately, so quitting never loses work. Returns total cases."""
    cases = load_cases()
    idx = next((i for i, c in enumerate(cases) if c.id == case.id), None)
    if idx is None:
        cases.append(case)
    else:
        cases[idx] = case
    save_cases(cases)
    return len(cases)


def describe(case: EvalCase) -> str:
    n2 = sum(1 for g in case.relevant.values() if g == 2)
    return f"{n2} exact, {len(case.relevant) - n2} acceptable"


# ===========================================================================
# Plan mode: work through evals/query_plan.jsonl
# ===========================================================================

async def run_plan(args: argparse.Namespace) -> None:
    plan = load_plan(args.plan_file)
    done = {c.id for c in load_cases()}
    todo: list[PlannedQuery] = [p for p in plan if args.relabel or p.id not in done]
    if args.only_tag:
        todo = [p for p in todo if set(args.only_tag) & set(p.tags)]
    if not todo:
        print(f"Nothing left to label: all {len(plan)} planned queries are in {DEFAULT_PATH.name}.")
        return

    print(f"{len(todo)} queries to label ({len(done)} already done). "
          "Enter grades, 's' to skip, 'q' to quit. Progress is saved after every query.")
    labeled = skipped = 0
    depth = min(max(args.depth, 1), 50)

    async with create_pool(max_size=4) as pool:
        for i, item in enumerate(todo, start=1):
            query = SearchQuery(text=item.query, filters=item.filters, limit=depth)
            print("\n" + "=" * 100)
            print(f"[{i}/{len(todo)}]  {item.id}   tags: {', '.join(item.tags) or '-'}")
            if item.notes:
                print(f"Guide: {item.notes}")
            print_query_header(query)
            items = await pool_candidates(pool, query)
            print_candidates(items)

            while True:
                try:
                    answer = input("\n> ").strip()
                except EOFError:
                    answer = "q"
                if answer.lower() in ("q", "quit"):
                    print(f"\nStopped. Labeled {labeled}, skipped {skipped}; "
                          f"{len(todo) - i + 1} left for next time.")
                    return
                if answer.lower() in ("", "s", "skip"):
                    skipped += 1
                    print("  skipped")
                    break
                try:
                    positions, extra = parse_answer(answer, len(items))
                except ValueError as e:
                    print(f"  {e}. Try again.")
                    continue
                case = build_case(item.id, query, items, positions, extra, item.tags, item.notes)
                if case is None:
                    print("  Nothing marked relevant. Enter grades, or 's' to skip.")
                    continue
                total = upsert_case(case)
                labeled += 1
                print(f"  saved: {describe(case)}  ({total} cases total)")
                break

    print(f"\nDone. Labeled {labeled}, skipped {skipped}.")


# ===========================================================================
# Single-query mode
# ===========================================================================

async def run_single(args: argparse.Namespace) -> None:
    case_id = args.id or slugify(args.text)
    if any(c.id == case_id for c in load_cases()) and not args.replace:
        sys.exit(f"Case {case_id!r} already exists. Use --replace to overwrite, or --id for a new one.")

    query = SearchQuery(text=args.text, filters=filters_from_args(args), limit=min(max(args.depth, 1), 50))
    async with create_pool(max_size=4) as pool:
        items = await pool_candidates(pool, query)
    print_query_header(query)
    print_candidates(items)

    try:
        also = parse_also(args.also)
        answer = args.grades
        if answer is None:
            print("\nRelevant? e.g. '1 2 5=1' (bare number = exact match, =1 = acceptable). Blank cancels.")
            answer = input("> ")
        positions, extra = parse_answer(answer, len(items))
    except ValueError as e:
        sys.exit(f"Not saved: {e}")

    case = build_case(case_id, query, items, positions, {**extra, **also}, args.tag, args.notes)
    if case is None:
        sys.exit("Nothing marked relevant; not saved.")
    total = upsert_case(case)
    print(f"\nSaved {case_id!r}: {describe(case)} ({total} cases in {DEFAULT_PATH.name}).")


async def main() -> None:
    ap = argparse.ArgumentParser(description="Label relevant products for eval queries.")
    ap.add_argument("text", nargs="?", help="a single query, as a shopper would type it")
    ap.add_argument("--plan", action="store_true", help=f"work through {PLAN_PATH.name}")
    ap.add_argument("--plan-file", default=PLAN_PATH, type=lambda p: __import__("pathlib").Path(p))
    ap.add_argument("--only-tag", nargs="+", default=[], help="plan mode: only queries with these tags")
    ap.add_argument("--relabel", action="store_true", help="plan mode: include queries already labeled")
    ap.add_argument("--id", help="case id (default: slug of the query)")
    ap.add_argument("--tag", nargs="+", default=[], help="e.g. exact-name vague-intent filtered collab")
    ap.add_argument("--notes", default="", help="what a good answer looks like, for future you")
    ap.add_argument("--depth", type=int, default=20, help="results taken from each mode (max 50)")
    ap.add_argument("--grades", help="answer the prompt non-interactively, e.g. '1 2 5=1'")
    ap.add_argument("--also", nargs="+", default=[], metavar="ID[=GRADE]",
                    help="relevant products none of the methods returned")
    ap.add_argument("--replace", action="store_true", help="overwrite an existing case with this id")
    add_filter_arguments(ap)
    args = ap.parse_args()

    if args.plan:
        await run_plan(args)
    elif args.text:
        await run_single(args)
    else:
        ap.error("give a query, or use --plan")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())