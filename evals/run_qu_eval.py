"""
Score query understanding on its own: does the LLM produce the right plan?

Retrieval metrics say whether results got better; this says whether the plans
are right, so an improvement can be traced to correct plans rather than luck.

    python -m evals.run_qu_eval
    python -m evals.run_qu_eval --only nl-black-sneakers typo-nupste

Each case in evals/qu_cases.jsonl lists only the fields it cares about:

    {"id": "...", "query": "...", "expect": {"categories": ["footwear"], "max_price": 150}}

Checks (values are compared after dropping anything not in the catalog, which
is what search actually uses):
    categories, subcategories, genders, brands,
    exclude_brands                                same set (case-insensitive); [] means "none"
    min_price, max_price                          equal number; null means "not set"
    intent, audience, on_sale_only, out_of_scope  equal
    search_text_contains                          every word appears in search_text
    search_text_excludes                          none of these words appears in search_text
    compare_items_count                           number of products to compare
    standalone_contains                           every word appears in standalone_query

A case can also carry "history": ["earlier message", ...] (the shopper's earlier turns).

Writes evals/results/qu_<timestamp>.md (commit it).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.db import create_pool
from app.query.understanding import PROMPT_VERSION, understand
from app.query.vocabulary import load_vocabulary
from evals.metrics import percentile

CASES_PATH = Path(__file__).parent / "qu_cases.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
SET_FIELDS = ("categories", "subcategories", "genders", "brands", "exclude_brands")
EXACT_FIELDS = ("intent", "audience", "on_sale_only", "out_of_scope")
PRICE_FIELDS = ("min_price", "max_price")


def load_qu_cases(path: Path = CASES_PATH) -> list[dict]:
    cases, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            c = json.loads(line)
            assert isinstance(c["id"], str) and isinstance(c["query"], str) and isinstance(c["expect"], dict)
        except (ValueError, KeyError, AssertionError) as e:
            raise ValueError(f"{path.name} line {n}: {e!r}") from None
        if c["id"] in seen:
            raise ValueError(f"{path.name} line {n}: duplicate id {c['id']!r}")
        seen.add(c["id"])
        cases.append(c)
    return cases


def check(expect: dict, got: dict) -> dict[str, tuple[bool, object, object]]:
    """field -> (passed, expected, got) for every field the case specifies."""
    out = {}
    for f, exp in expect.items():
        if f in SET_FIELDS:
            g = sorted({str(x).lower() for x in got.get(f, [])})
            out[f] = (g == sorted({str(x).lower() for x in exp}), sorted(exp), g)
        elif f in PRICE_FIELDS:
            g = got.get(f)
            ok = (g is None and exp is None) or (g is not None and exp is not None and abs(float(g) - float(exp)) < 0.01)
            out[f] = (ok, exp, g)
        elif f in EXACT_FIELDS:
            out[f] = (got.get(f) == exp, exp, got.get(f))
        elif f == "compare_items_count":
            n = len(got.get("compare_items", []))
            out[f] = (n == exp, exp, got.get("compare_items", []))
        elif f == "standalone_contains":
            text = got.get("standalone_query", "").lower()
            out[f] = (all(w.lower() in text for w in exp), exp, got.get("standalone_query", ""))
        elif f == "search_text_contains":
            text = got.get("search_text", "").lower()
            missing = [w for w in exp if w.lower() not in text]
            out[f] = (not missing, exp, got.get("search_text", ""))
        elif f == "search_text_excludes":
            text = got.get("search_text", "").lower()
            found = [w for w in exp if w.lower() in text]
            out[f] = (not found, exp, got.get("search_text", ""))
        else:
            raise ValueError(f"unknown expectation {f!r}")
    return out


async def run(cases: list[dict]) -> list[dict]:
    rows = []
    async with create_pool(max_size=2) as pool:
        vocab = await load_vocabulary(pool)
        for i, c in enumerate(cases, start=1):
            try:
                u = await understand(c["query"], vocab, c.get("history"))
            except Exception as e:     # noqa: BLE001 - a failed call is a failed case, not a crash
                rows.append({"case": c, "error": f"{type(e).__name__}: {e}"[:300], "checks": {}})
                continue
            plan = u.plan.model_dump(mode="json")
            cats, subs, brands, dropped = vocab.clean(categories=plan["categories"],
                                                      subcategories=plan["subcategories"], brands=plan["brands"])
            _, _, excluded, _ = vocab.clean(categories=[], subcategories=[], brands=plan["exclude_brands"])
            got = {**plan, "categories": cats, "subcategories": subs, "brands": brands, "exclude_brands": excluded}
            rows.append({"case": c, "plan": plan, "dropped": dropped, "checks": check(c["expect"], got),
                         "latency_ms": u.latency_ms, "tokens": (u.input_tokens, u.output_tokens)})
            print(f"\r  {i}/{len(cases)} cases", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return rows


def report(rows: list[dict]) -> str:
    s = get_settings()
    by_field: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        for f, (ok, _, _) in r["checks"].items():
            by_field[f].append(ok)
    passed = [r for r in rows if r["checks"] and all(ok for ok, _, _ in r["checks"].values())]
    lat = [r["latency_ms"] for r in rows if "latency_ms" in r]
    tin = sum(r.get("tokens", (0, 0))[0] for r in rows)
    tout = sum(r.get("tokens", (0, 0))[1] for r in rows)

    out = [
        f"# Query-understanding eval, {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"{len(rows)} cases. Model {s.query_model}, prompt {PROMPT_VERSION}.",
        "",
        f"**Cases fully correct: {len(passed)}/{len(rows)}** "
        f"({len(passed) / max(len(rows), 1):.0%}). Errors: {sum('error' in r for r in rows)}.",
        "",
        f"Latency p50 {percentile(lat, 50):.0f} ms, p95 {percentile(lat, 95):.0f} ms. "
        f"Tokens {tin:,} in / {tout:,} out ({tin // max(len(lat), 1):,} / {tout // max(len(lat), 1):,} per call).",
        "",
        "| Field | Correct | Accuracy |",
        "|---|---|---|",
    ]
    for f in sorted(by_field, key=lambda f: (sum(by_field[f]) / len(by_field[f]), f)):
        v = by_field[f]
        out.append(f"| {f} | {sum(v)}/{len(v)} | {sum(v) / len(v):.0%} |")

    failures = [r for r in rows if r not in passed]
    if failures:
        out += ["", "## Misses", ""]
        for r in failures:
            c = r["case"]
            if "error" in r:
                out.append(f"- `{c['id']}` ({c['query']!r}): **error** {r['error']}")
                continue
            bad = [f"{f}: expected `{exp}`, got `{got}`" for f, (ok, exp, got) in r["checks"].items() if not ok]
            extra = f" (dropped as not in catalog: {r['dropped']})" if r.get("dropped") else ""
            out.append(f"- `{c['id']}` ({c['query']!r}): " + "; ".join(bad) + extra)
    return "\n".join(out) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser(description="Score the query-understanding plans.")
    ap.add_argument("--only", nargs="+", default=[], help="case ids to run")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    cases = load_qu_cases()
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    rows = await run(cases)
    text = report(rows)
    print(text)
    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"qu_{datetime.now():%Y%m%d-%H%M%S}.md"
        path.write_text(text, encoding="utf-8")
        print(f"Saved {path}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
