"""
Hand-label generated answers pass / fail, to measure how far the judge can be trusted.

A judge is only useful if it agrees with a careful human. Label about 30
answers (a mix of good and bad) BEFORE reading the judge's verdicts, so its
opinion doesn't steer yours; judge.py then reports agreement and Cohen's
kappa on the answers you both graded.

    python -m evals.label_answers evals/results/gen_20261016-141220.jsonl
    python -m evals.label_answers evals/results/gen_20261016-141220.jsonl --only rec-nuptse
    python -m evals.label_answers evals/results/gen_20261016-141220.jsonl --relabel

Ask one question: would you be happy if a shopper got this answer? Fail it
for a claim the product blocks don't support, a pick that doesn't fit, the
wrong kind of answer, or a message that doesn't answer what was asked.

At the prompt:
    p [note]     pass
    f [note]     fail (a note on why helps when you read disagreements later)
    a            show every product block, not only the picked ones
    s            skip
    q            quit (labels so far are saved)

Labels go to evals/answer_labels.jsonl (commit it), keyed by the answer's
hash: a re-generated answer needs a new label. Fixed replies are not shown.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from evals.answers import answer_hash, load_answers, product_blocks, render_answer, render_request

LABELS_PATH = Path(__file__).parent / "answer_labels.jsonl"


def load_labels(path: Path = LABELS_PATH) -> dict[str, dict]:
    """answer_hash -> label row. A later line for the same answer replaces an earlier one."""
    if not path.exists():
        return {}
    out = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            assert row["label"] in ("pass", "fail") and isinstance(row["answer_hash"], str)
        except (ValueError, KeyError, AssertionError) as e:
            raise ValueError(f"{path.name} line {n}: {e!r}") from None
        out[row["answer_hash"]] = row
    return out


def parse_input(text: str) -> tuple[str, str]:
    """'f wrong colour' -> ('fail', 'wrong colour'). Raises ValueError on anything else."""
    cmd, _, note = text.strip().partition(" ")
    cmd = cmd.lower()
    if cmd in ("p", "pass"):
        return "pass", note.strip()
    if cmd in ("f", "fail"):
        return "fail", note.strip()
    if cmd in ("a", "s", "q") and not note.strip():
        return cmd, ""
    raise ValueError(f"didn't understand {text!r}")


def show(row: dict, all_products: bool = False) -> str:
    blocks = product_blocks(row.get("context", ""))
    picked = {p["product_id"] for p in row.get("picks", [])}
    lines = ["", "=" * 78, f"{row['case_id']}", "-" * 78, render_request(row), "", render_answer(row), ""]
    for pid, body in blocks.items():
        if all_products or pid in picked:
            lines += [f"[{pid}]" + (" (picked)" if pid in picked else ""), body, ""]
    others = [pid for pid in blocks if pid not in picked]
    if others and not all_products:
        titles = [next((l[7:] for l in blocks[pid].splitlines() if l.startswith("Title: ")), "?") for pid in others]
        lines.append(f"Not picked ({len(others)}): " + "; ".join(titles) + "   [a] to show their blocks")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand-label generated answers.")
    ap.add_argument("answers", help="evals/results/gen_<stamp>.jsonl")
    ap.add_argument("--only", nargs="+", default=[], help="case ids to label")
    ap.add_argument("--relabel", action="store_true", help="also show answers you already labeled")
    args = ap.parse_args()

    labels = load_labels()
    rows = [r for r in load_answers(Path(args.answers)) if r.get("source") != "template"]
    if args.only:
        rows = [r for r in rows if r["case_id"] in set(args.only)]
    todo = [r for r in rows if args.relabel or answer_hash(r) not in labels]
    print(f"{len(todo)} answers to label ({len(rows) - len(todo)} already labeled). "
          f"{len(labels)} labels in {LABELS_PATH.name}.")

    done = 0
    for row in todo:
        print(show(row))
        while True:
            try:
                label, note = parse_input(input("\np / f [note], a, s, q > "))
            except ValueError as e:
                print(e)
                continue
            except EOFError:
                label, note = "q", ""
            if label == "a":
                print(show(row, all_products=True))
                continue
            break
        if label == "q":
            break
        if label == "s":
            continue
        with LABELS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"answer_hash": answer_hash(row), "case_id": row["case_id"], "label": label,
                                 "note": note, "answers_file": Path(args.answers).name,
                                 "labeled_at": datetime.now().isoformat(timespec="seconds")},
                                ensure_ascii=False) + "\n")
        done += 1
    print(f"\nSaved {done} labels to {LABELS_PATH}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nStopped; labels so far are saved.")