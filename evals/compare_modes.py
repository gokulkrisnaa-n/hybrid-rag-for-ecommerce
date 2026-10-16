"""
Per-query comparison of two modes in one eval run, worst losses first.

    python -m evals.compare_modes                       # latest run, hybrid vs hybrid+qu
    python -m evals.compare_modes --a hybrid --b dense
    python -m evals.compare_modes evals/results/retrieval_..._qu-v2.csv --top 15

For hybrid+qu rows it also prints the plan's search text and the filters it
inferred, which is usually enough to see why a query got worse.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def latest_csv() -> Path:
    files = sorted(RESULTS_DIR.glob("retrieval_*.csv"))
    if not files:
        raise SystemExit(f"No retrieval_*.csv in {RESULTS_DIR}; run evals.run_retrieval_eval first.")
    return files[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two modes query by query.")
    ap.add_argument("csv", nargs="?", type=Path, help="per-query CSV (default: the latest run)")
    ap.add_argument("--a", default="hybrid", help="baseline mode")
    ap.add_argument("--b", default="hybrid+qu", help="mode to compare")
    ap.add_argument("--top", type=int, default=12, help="how many losses and gains to show")
    args = ap.parse_args()

    path = args.csv or latest_csv()
    rows: dict[tuple[str, str], dict] = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        rows[(r["case_id"], r["mode"])] = r

    cases = sorted({cid for cid, _ in rows})
    diffs = []
    for cid in cases:
        a, b = rows.get((cid, args.a)), rows.get((cid, args.b))
        if a and b:
            diffs.append((float(b["ndcg"]) - float(a["ndcg"]), cid, a, b))
    if not diffs:
        raise SystemExit(f"{path.name} has no cases with both {args.a!r} and {args.b!r}.")

    def show(title: str, items: list) -> None:
        print(f"\n{title}")
        for d, cid, a, b in items:
            line = f"  {d:+.2f}  {cid:<34} {float(a['ndcg']):.2f} -> {float(b['ndcg']):.2f}"
            qu = json.loads(b.get("debug") or "{}").get("qu", {})
            if qu.get("used"):
                inferred = {k: v for k, v in (qu.get("inferred_filters") or {}).items()}
                line += f"\n         text: {qu.get('search_text')!r}  filters: {inferred or '-'}"
                if qu.get("relaxed"):
                    line += "  (relaxed)"
            print(line)

    diffs.sort(key=lambda x: (x[0], x[1]))
    worse = [x for x in diffs if x[0] < -0.005]
    better = [x for x in reversed(diffs) if x[0] > 0.005]
    print(f"{path.name}: {args.b} vs {args.a} on {len(diffs)} queries — "
          f"{len(better)} better, {len(worse)} worse, {len(diffs) - len(better) - len(worse)} same.")
    show(f"Worst losses ({args.b} below {args.a}):", worse[: args.top])
    show(f"Biggest gains ({args.b} above {args.a}):", better[: args.top])


if __name__ == "__main__":
    main()
