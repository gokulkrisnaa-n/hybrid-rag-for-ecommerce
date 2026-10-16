"""
Generation eval, step 1: rule checks on full chat answers.

Runs every case in evals/gen_cases.jsonl through the real pipeline (query
understanding, hybrid search, generation) and checks what code can check
exactly, with no judge model:

    kind            answer kind is one of the expected kinds
    no_fallback     no fallback to "closest matches" because generation failed or gave no valid picks
                    (deliberate fixed replies, e.g. out of scope, are fine)
    picks           number of picked products within min_picks / max_picks
    valid_ids       no picked product outside the retrieved set
    no_price_text   no price, dollar amount or "% off" written in the text
    compare_table   a "compare" answer has a table with at least two products
    filters         every pick satisfies the case's filters (price, gender, sale, in stock)
    mention         required words appear in the message or reasons
    not_mention     forbidden words don't

and, for cases linked to a retrieval ground-truth case, pick precision: the
share of picked products graded relevant (grade 1 or 2) in
evals/ground_truth.jsonl. Unlabeled products count as not relevant, so this
is a lower bound.

    python -m evals.run_generation_eval
    python -m evals.run_generation_eval --only compare-samba-gazelle nomatch-tuxedo
    python -m evals.run_generation_eval --tag follow-up safety

Writes evals/results/gen_<timestamp>.md (commit it) and gen_<timestamp>.jsonl
(every answer with the product context the model saw; input for the judge in
step 2; gitignored).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from app.chat.service import chat
from app.core import llm
from app.core.config import get_settings
from app.core.db import create_pool
from app.generation.prompt import PROMPT_VERSION as GEN_PROMPT
from app.generation.schemas import ChatAnswer, Turn
from app.query.understanding import PROMPT_VERSION as QU_PROMPT
from app.schemas.query_models import SearchFilters
from evals.dataset import load_cases
from evals.metrics import percentile

CASES_PATH = Path(__file__).parent / "gen_cases.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
CHECKS = ("kind", "no_fallback", "picks", "valid_ids", "no_price_text", "compare_table",
          "filters", "mention", "not_mention")
EXPECT_KEYS = {"kind", "source", "min_picks", "max_picks", "mention", "not_mention"}


def load_gen_cases(path: Path = CASES_PATH) -> list[dict]:
    cases, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            c = json.loads(line)
            assert isinstance(c["id"], str) and isinstance(c["message"], str)
            assert set(c["expect"]) <= EXPECT_KEYS, f"unknown expect keys {set(c['expect']) - EXPECT_KEYS}"
            SearchFilters.model_validate(c.get("filters", {}))
            [Turn.model_validate(t) for t in c.get("history", [])]
        except (ValueError, KeyError, AssertionError) as e:
            raise ValueError(f"{path.name} line {n}: {e!r}") from None
        if c["id"] in seen:
            raise ValueError(f"{path.name} line {n}: duplicate id {c['id']!r}")
        seen.add(c["id"])
        cases.append(c)
    return cases


def _pick_violates(card, f: SearchFilters) -> str | None:
    if f.max_price is not None and card.price_min > f.max_price:
        return f"price {card.price_min} > {f.max_price}"
    if f.min_price is not None and card.price_max < f.min_price:
        return f"price {card.price_max} < {f.min_price}"
    if f.genders and card.gender not in f.genders:
        return f"gender {card.gender} not in {f.genders}"
    if f.on_sale_only and not card.on_sale:
        return "not on sale"
    if f.in_stock_only and not card.in_stock:
        return "sold out"
    return None


def check_answer(case: dict, a: ChatAnswer) -> dict[str, tuple[bool, str]]:
    """check -> (passed, detail). Checks that don't apply to the case are left out."""
    exp = case["expect"]
    f = SearchFilters.model_validate(case.get("filters", {}))
    text = " ".join([a.message] + [p.reason for p in a.products]).lower()
    n = len(a.products)
    out: dict[str, tuple[bool, str]] = {}

    out["kind"] = (a.kind in exp["kind"], f"{a.kind} (expected {'/'.join(exp['kind'])})")
    # A fallback is generation failing (or producing no usable picks) and the shopper getting
    # "here are the closest matches". A deliberate fixed reply (out of scope, no results) is not one.
    fell_back = a.kind == "error" or any(w.startswith(("generation failed", "no valid picks")) for w in a.warnings)
    if "source" in exp:
        out["no_fallback"] = (a.source == exp["source"] and not fell_back, a.source)
    else:
        out["no_fallback"] = (not fell_back, "ok" if not fell_back else f"fallback: {a.warnings[:1]}")
    lo, hi = exp.get("min_picks", 0), exp.get("max_picks", get_settings().generation_max_picks)
    out["picks"] = (lo <= n <= hi, f"{n} picks (expected {lo}-{hi})")
    unknown = [w for w in a.warnings if "unknown product_id" in w]
    out["valid_ids"] = (not unknown, "; ".join(unknown) or "ok")
    out["no_price_text"] = (not any("price or discount" in w for w in a.warnings), "ok")
    if a.kind == "compare":
        rows = len(a.comparison.rows) if a.comparison else 0
        out["compare_table"] = (rows >= 2, f"{rows} rows")
    bad = [f"{p.product.product_id}: {v}" for p in a.products if (v := _pick_violates(p.product, f))]
    out["filters"] = (not bad, "; ".join(bad) or "ok")
    if exp.get("mention"):
        missing = [w for w in exp["mention"] if w.lower() not in text]
        out["mention"] = (not missing, f"missing {missing}" if missing else "ok")
    if exp.get("not_mention"):
        found = [w for w in exp["not_mention"] if w.lower() in text]
        out["not_mention"] = (not found, f"found {found}" if found else "ok")
    return out


def pick_precision(a: ChatAnswer, relevant: dict[int, int]) -> tuple[float, bool] | None:
    if not a.products:
        return None
    ids = [p.product.product_id for p in a.products]
    hits = [relevant.get(i, 0) > 0 for i in ids]
    return sum(hits) / len(hits), hits[0]


async def run(cases: list[dict]) -> list[dict]:
    truth = {c.id: c.relevant for c in load_cases()}
    rows = []
    async with create_pool(max_size=4) as pool:
        for i, c in enumerate(cases, start=1):
            llm.clear_query_cache()
            history = [Turn.model_validate(t) for t in c.get("history", [])]
            filters = SearchFilters.model_validate(c.get("filters", {}))
            try:
                a = await chat(pool, c["message"], history, filters, include_context=True)
            except Exception as e:     # noqa: BLE001 - a crash is a failed case, not a failed run
                rows.append({"case": c, "error": f"{type(e).__name__}: {e}"[:300]})
                continue
            row = {"case": c, "answer": a, "checks": check_answer(c, a)}
            if c.get("relevance_case") in truth:
                row["precision"] = pick_precision(a, truth[c["relevance_case"]])
            rows.append(row)
            print(f"\r  {i}/{len(cases)} cases", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return rows


def report(rows: list[dict]) -> str:
    s = get_settings()
    ok_rows = [r for r in rows if "answer" in r]
    by_check: dict[str, list[bool]] = defaultdict(list)
    for r in ok_rows:
        for name, (passed, _) in r["checks"].items():
            by_check[name].append(passed)
    passed = [r for r in ok_rows if all(p for p, _ in r["checks"].values())]
    prec = [r["precision"] for r in ok_rows if r.get("precision")]
    t = lambda k: [r["answer"].timings_ms[k] for r in ok_rows if k in r["answer"].timings_ms]
    tok = [r["answer"].debug.get("tokens", {}) for r in ok_rows]
    kinds = defaultdict(int)
    for r in ok_rows:
        kinds[r["answer"].kind] += 1

    out = [
        f"# Generation eval, {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"{len(rows)} cases. Generation {s.generation_model} (prompt {GEN_PROMPT}, reasoning "
        f"{s.generation_reasoning_effort or 'default'}); query understanding {s.query_model} ({QU_PROMPT}); "
        f"{s.generation_candidates} candidates, up to {s.generation_max_picks} picks.",
        "",
        f"**Cases passing every check: {len(passed)}/{len(rows)}** "
        f"({len(passed) / max(len(rows), 1):.0%}). Crashed: {len(rows) - len(ok_rows)}.",
        "",
        "| Check | Passed | Rate |",
        "|---|---|---|",
    ]
    for name in CHECKS:
        v = by_check.get(name)
        if v:
            out.append(f"| {name} | {sum(v)}/{len(v)} | {sum(v) / len(v):.0%} |")

    if prec:
        out += ["", f"**Pick precision** (picks graded relevant in the retrieval ground truth, a lower bound): "
                    f"mean {statistics.fmean(p for p, _ in prec):.2f}; top pick relevant in "
                    f"{sum(top for _, top in prec)}/{len(prec)} cases."]

    out += ["", "**Answer kinds:** " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items())), ""]
    lat = [("search", t("search")), ("generate", t("generate")), ("total", t("total"))]
    out.append("**Latency:** " + "; ".join(f"{k} p50 {percentile(v, 50):.0f} ms / p95 {percentile(v, 95):.0f} ms"
                                           for k, v in lat if v))
    tin, tout = sum(x.get("input", 0) for x in tok), sum(x.get("output", 0) for x in tok)
    n_llm = sum(1 for x in tok if x)
    if n_llm:
        out.append(f"\n**Generation tokens:** {tin:,} in / {tout:,} out ({tin // n_llm:,} / {tout // n_llm:,} per answer).")

    failures = [r for r in rows if r not in passed]
    if failures:
        out += ["", "## Failures", ""]
        for r in failures:
            c = r["case"]
            if "error" in r:
                out.append(f"- `{c['id']}`: **crashed** {r['error']}")
                continue
            bad = [f"{name}: {detail}" for name, (p, detail) in r["checks"].items() if not p]
            out.append(f"- `{c['id']}` ({c['message']!r}): " + "; ".join(bad))
            out.append(f"  - answer [{r['answer'].kind}]: {r['answer'].message[:220]}")
    return "\n".join(out) + "\n"


def dump_answers(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            if "answer" not in r:
                continue
            a: ChatAnswer = r["answer"]
            fh.write(json.dumps({
                "case_id": r["case"]["id"],
                "message": r["case"]["message"],
                "history": r["case"].get("history", []),
                "filters": r["case"].get("filters", {}),
                "kind": a.kind,
                "source": a.source,
                "answer_message": a.message,
                "picks": [{"product_id": p.product.product_id, "title": p.product.title, "reason": p.reason}
                          for p in a.products],
                "comparison": a.comparison.model_dump(mode="json") if a.comparison else None,
                "context": a.debug.get("context", ""),
                "checks": {k: v[0] for k, v in r["checks"].items()},
            }, default=str) + "\n")


async def main() -> None:
    ap = argparse.ArgumentParser(description="Rule-check generated answers.")
    ap.add_argument("--only", nargs="+", default=[], help="case ids to run")
    ap.add_argument("--tag", nargs="+", default=[], help="only cases with any of these tags")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    cases = load_gen_cases()
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    if args.tag:
        cases = [c for c in cases if set(args.tag) & set(c.get("tags", []))]
    if not cases:
        sys.exit("No cases selected.")

    rows = await run(cases)
    text = report(rows)
    print(text)
    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        md, js = RESULTS_DIR / f"gen_{stamp}.md", RESULTS_DIR / f"gen_{stamp}.jsonl"
        md.write_text(text, encoding="utf-8")
        dump_answers(rows, js)
        print(f"Saved {md.name} and {js.name} (answers with context, for the judge step).")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
