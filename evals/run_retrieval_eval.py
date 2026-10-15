"""
Score retrieval: run every case in evals/ground_truth.jsonl through each mode
and compare.

    python -m evals.run_retrieval_eval
    python -m evals.run_retrieval_eval --k 5 --tag vague-intent

Prints a summary per mode and per tag, the queries each mode does worst on,
and writes two files to evals/results/:
    retrieval_<timestamp>.md   the summary (commit this; it's your evidence)
    retrieval_<timestamp>.csv  one row per query per mode (for digging in)

Query embeddings are cached in memory, so dense and hybrid share one OpenAI
call per query, and a full run costs a fraction of a cent.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.db import create_pool
from app.retrieval.hybrid import hybrid_search
from evals.dataset import DEFAULT_PATH, EvalCase, load_cases
from evals.metrics import mrr_at_k, ndcg_at_k, percentile, recall_at_k

MODES = ("bm25", "dense", "hybrid")
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class Row:
    case: EvalCase
    mode: str
    recall: float
    mrr: float
    ndcg: float
    latency_ms: float
    ranked: list[int]


async def evaluate(cases: list[EvalCase], k: int, modes: tuple[str, ...] = MODES) -> list[Row]:
    rows: list[Row] = []
    async with create_pool(max_size=4) as pool:
        for i, case in enumerate(cases, start=1):
            query = case.to_search_query(limit=k)
            for mode in modes:
                res = await hybrid_search(pool, query, mode)
                ranked = [r.product.product_id for r in res.results]
                rows.append(Row(
                    case, mode,
                    recall_at_k(ranked, case.relevant, k),
                    mrr_at_k(ranked, case.relevant, k),
                    ndcg_at_k(ranked, case.relevant, k),
                    res.timings_ms.get("total", 0.0),
                    ranked,
                ))
            print(f"\r  {i}/{len(cases)} queries", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return rows


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def summarize(rows: list[Row], k: int, modes: tuple[str, ...]) -> str:
    by_mode: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_mode[r.mode].append(r)
    n_cases = len({r.case.id for r in rows})
    s = get_settings()

    out = [
        f"# Retrieval eval, {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"{n_cases} queries, k = {k}. Embeddings: {s.embedding_model} ({s.embedding_dimensions} dims); "
        f"RRF k = {s.rrf_k}; {s.retrieval_candidates} candidates per method.",
        "",
        f"| Mode | Recall@{k} | MRR | nDCG@{k} | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
    ]
    for mode in modes:
        rs = by_mode[mode]
        lat = [r.latency_ms for r in rs]
        out.append(
            f"| {mode} | {_mean([r.recall for r in rs]):.3f} | {_mean([r.mrr for r in rs]):.3f} | "
            f"{_mean([r.ndcg for r in rs]):.3f} | {percentile(lat, 50):.0f} | {percentile(lat, 95):.0f} |"
        )

    tags = sorted({t for r in rows for t in r.case.tags})
    if tags:
        out += ["", f"## nDCG@{k} by query type", "",
                "| Tag | Queries | " + " | ".join(modes) + " |",
                "|---|---|" + "---|" * len(modes)]
        for tag in tags:
            n = len({r.case.id for r in rows if tag in r.case.tags})
            cells = [f"{_mean([r.ndcg for r in by_mode[m] if tag in r.case.tags]):.3f}" for m in modes]
            top = max(cells, key=float)
            if len(cells) > 1 and cells.count(top) == 1:      # bold a clear winner only, not a tie
                cells[cells.index(top)] = f"**{top}**"
            out.append(f"| {tag} | {n} | " + " | ".join(cells) + " |")

    out += ["", f"## Weakest queries per mode (lowest nDCG@{k})", ""]
    for mode in modes:
        worst = sorted(by_mode[mode], key=lambda r: (r.ndcg, r.case.id))[:5]
        out.append(f"**{mode}**: " + ", ".join(f"`{r.case.id}` ({r.ndcg:.2f})" for r in worst))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def write_csv(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "query", "tags", "mode", "recall", "mrr", "ndcg", "latency_ms",
                    "relevant_found", "relevant_total", "top_ids"])
        for r in rows:
            rel = {p for p, g in r.case.relevant.items() if g > 0}
            w.writerow([r.case.id, r.case.query, " ".join(r.case.tags), r.mode,
                        f"{r.recall:.4f}", f"{r.mrr:.4f}", f"{r.ndcg:.4f}", f"{r.latency_ms:.0f}",
                        len(rel & set(r.ranked)), len(rel), " ".join(map(str, r.ranked))])


async def main() -> None:
    ap = argparse.ArgumentParser(description="Score BM25, vector and hybrid retrieval on the eval set.")
    ap.add_argument("--k", type=int, default=10, help="cutoff for every metric (max 50)")
    ap.add_argument("--tag", nargs="+", default=[], help="only cases with any of these tags")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--no-save", action="store_true", help="print only; don't write result files")
    args = ap.parse_args()
    if not 1 <= args.k <= 50:
        ap.error("--k must be between 1 and 50")

    cases = load_cases()
    if args.tag:
        cases = [c for c in cases if set(args.tag) & set(c.tags)]
    if not cases:
        sys.exit(f"No cases to run. Add some with: python -m evals.label \"your query\"  ({DEFAULT_PATH})")

    modes = tuple(m for m in MODES if m in args.modes)
    rows = await evaluate(cases, args.k, modes)
    report = summarize(rows, args.k, modes)
    print(report)

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        md, cs = RESULTS_DIR / f"retrieval_{stamp}.md", RESULTS_DIR / f"retrieval_{stamp}.csv"
        md.write_text(report, encoding="utf-8")
        write_csv(rows, cs)
        print(f"Saved {md.relative_to(Path.cwd()) if md.is_relative_to(Path.cwd()) else md} and the per-query CSV.")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())