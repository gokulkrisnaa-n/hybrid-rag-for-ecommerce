"""
Latency benchmark: where the time goes in a chat turn, and what the shopper waits for.

Runs the cases in evals/gen_cases.jsonl through the real pipeline (database,
OpenAI) as streamed turns, one at a time, with the plan and embedding caches
cleared before each case so every turn pays full price. Reports p50 / p95 per
step and the time to the first words.

    python -m evals.run_latency --label baseline
    PREFETCH_QUERY_EMBEDDING=false python -m evals.run_latency --label no-prefetch
    QUERY_REASONING_EFFORT=minimal python -m evals.run_latency --label qu-minimal
    python -m evals.run_latency --only rec-nuptse nl-black-sneakers --repeat 3

Compare two runs by their labels; settings that change latency are printed in
each report. With tracing on, every case is one trace tagged "benchmark" (run
with TRACING_ENVIRONMENT=benchmark to keep them apart from real traffic).

Writes evals/results/latency_<stamp>_<label>.md (commit it).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from app.chat.service import chat_stream
from app.core import llm, tracing
from app.core.config import get_settings
from app.core.db import create_pool
from app.generation.prompt import PROMPT_VERSION as GEN_PROMPT
from app.generation.schemas import Turn
from app.query import understanding
from app.query.understanding import PROMPT_VERSION as QU_PROMPT
from app.schemas.query_models import SearchFilters
from evals.metrics import percentile
from evals.run_generation_eval import load_gen_cases

RESULTS_DIR = Path(__file__).parent / "results"

# (label, where the number comes from)
STEPS = [
    ("understand", "search_timings.understand"),
    ("embed (wait)", "search_timings.embed_query"),
    ("bm25", "search_timings.bm25_search"),
    ("dense", "search_timings.dense_search"),
    ("search total", "timings.search"),
    ("load details", "timings.context"),
    ("generate", "timings.generate"),
    ("first words", "timings.first_token"),
    ("total", "timings.total"),
]


def _value(row: dict, path: str) -> float | None:
    source, key = path.split(".", 1)
    return (row[source] or {}).get(key)


async def run(cases: list[dict], repeat: int, label: str) -> list[dict]:
    rows = []
    async with create_pool(max_size=4) as pool:
        total = len(cases) * repeat
        n = 0
        for _ in range(repeat):
            for c in cases:
                n += 1
                llm.clear_query_cache()
                understanding.clear_plan_cache()
                history = [Turn.model_validate(t) for t in c.get("history", [])]
                filters = SearchFilters.model_validate(c.get("filters", {}))

                async def emit(event, data):
                    pass
                started = time.perf_counter()
                try:
                    with tracing.request_trace("chat", request_id=uuid.uuid4().hex,
                                               input={"message": tracing.mask_pii(c["message"])},
                                               tags=["benchmark", label, f"qu:{QU_PROMPT}", f"gen:{GEN_PROMPT}"]):
                        answer = await chat_stream(pool, c["message"], history, filters, emit=emit)
                except Exception as e:     # noqa: BLE001 - one failure is one row, not a crashed run
                    rows.append({"case": c["id"], "error": f"{type(e).__name__}: {e}"[:200]})
                    continue
                rows.append({
                    "case": c["id"],
                    "kind": answer.kind,
                    "wall_ms": (time.perf_counter() - started) * 1000,
                    "timings": answer.timings_ms,
                    "search_timings": answer.debug.get("search_timings", {}),
                    "qu_used": (answer.debug.get("qu") or {}).get("used"),
                })
                print(f"\r  {n}/{total} turns", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    tracing.flush()
    return rows


def report(rows: list[dict], label: str) -> str:
    s = get_settings()
    ok = [r for r in rows if "error" not in r]
    out = [
        f"# Latency benchmark, {datetime.now():%Y-%m-%d %H:%M} ({label})",
        "",
        f"{len(ok)} turns ({len(rows) - len(ok)} failed). Streamed; plan and embedding caches cleared before "
        f"each turn. Query understanding {s.query_model} ({QU_PROMPT}, reasoning "
        f"{s.query_reasoning_effort or 'default'}); generation {s.generation_model} ({GEN_PROMPT}, reasoning "
        f"{s.generation_reasoning_effort or 'default'}); embedding prefetch "
        f"{'on' if s.prefetch_query_embedding else 'off'}; {s.generation_candidates} candidates.",
        "",
        "| Step | Turns | p50 ms | p95 ms | Max ms |",
        "|---|---|---|---|---|",
    ]
    for name, path in STEPS:
        vals = [v for r in ok if (v := _value(r, path)) is not None]
        if vals:
            out.append(f"| {name} | {len(vals)} | {percentile(vals, 50):,.0f} | {percentile(vals, 95):,.0f} "
                       f"| {max(vals):,.0f} |")
    out += ["", "`embed (wait)` is how long retrieval waited for the query vector: near 0 when the "
                "prefetch finished while query understanding ran. `first words`: from the start of the turn to "
                "the first streamed text (turns answered by the model only; fixed replies have none)."]

    slow = sorted(ok, key=lambda r: r["timings"].get("total", 0), reverse=True)[:5]
    if slow:
        out += ["", "**Slowest turns:** " + "; ".join(
            f"`{r['case']}` {r['timings'].get('total', 0):,.0f} ms ({r['kind']})" for r in slow)]
    failed = [r for r in rows if "error" in r]
    if failed:
        out += ["", "**Failed:** " + "; ".join(f"`{r['case']}`: {r['error']}" for r in failed)]
    return "\n".join(out) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser(description="Measure chat-turn latency per step.")
    ap.add_argument("--label", default="run", help="name for this run, e.g. baseline, no-prefetch")
    ap.add_argument("--only", nargs="+", default=[], help="case ids to run")
    ap.add_argument("--repeat", type=int, default=1, help="run every case this many times")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    cases = load_gen_cases()
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]
    if not cases:
        sys.exit("No cases selected.")
    rows = await run(cases, max(1, args.repeat), args.label)
    text = report(rows, args.label)
    print(text)
    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / f"latency_{datetime.now():%Y%m%d-%H%M%S}_{args.label}.md"
        path.write_text(text, encoding="utf-8")
        print(f"Saved {path.name}.")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
