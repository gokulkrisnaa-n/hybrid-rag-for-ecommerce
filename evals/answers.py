"""
Saved chat answers, shared by the judge and the hand-labeling tool.

run_generation_eval writes evals/results/gen_<stamp>.jsonl: every answer with
the request and the product blocks the model saw. This module reads that
file, renders an answer as text (the judge and you see the same thing), and
gives each answer a stable id.

The id is a hash of the request, the product context and the answer, so a
hand label always belongs to the exact answer you graded. Re-running the
generation eval gives new answers and new ids; labels for old answers are
simply not matched.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_BLOCK = re.compile(r'<product id="(\d+)">\n(.*?)\n</product>', re.S)


def load_answers(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            assert isinstance(row["case_id"], str) and isinstance(row["answer_message"], str)
        except (ValueError, KeyError, AssertionError) as e:
            raise ValueError(f"{Path(path).name} line {n}: {e!r}") from None
        rows.append(row)
    return rows


def answer_hash(row: dict) -> str:
    """12 hex characters; changes whenever the request, the context or the answer changes."""
    keys = ("message", "history", "filters", "context", "kind", "answer_message", "picks", "comparison")
    blob = json.dumps({k: row.get(k) for k in keys}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def product_blocks(context: str) -> dict[int, str]:
    """product_id -> the block's lines (without the tags)."""
    return {int(pid): body for pid, body in _BLOCK.findall(context or "")}


def render_request(row: dict) -> str:
    lines = []
    if row.get("history"):
        lines.append("Earlier conversation:")
        lines += [f"  {t['role']}: {t['content']}" for t in row["history"]]
    if row.get("filters"):
        lines.append("Filters the shopper set: " + "; ".join(f"{k}: {v}" for k, v in row["filters"].items()))
    lines.append(f"Shopper message: {row['message']}")
    return "\n".join(lines)


def render_answer(row: dict) -> str:
    lines = [f"Kind: {row['kind']}", f"Message: {row['answer_message']}"]
    if row.get("picks"):
        lines.append("Picks (best first):")
        lines += [f"  {i}. [{p['product_id']}] {p['title']}: {p['reason']}"
                  for i, p in enumerate(row["picks"], start=1)]
    else:
        lines.append("Picks: none")
    table = row.get("comparison")
    if table and table.get("rows"):
        lines.append("Comparison table:")
        lines.append("  | Product | " + " | ".join(table["columns"]) + " |")
        lines += [f"  | {r['title']} | " + " | ".join(r["values"]) + " |" for r in table["rows"]]
    return "\n".join(lines)
