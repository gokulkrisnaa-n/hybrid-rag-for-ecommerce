"""
Embed the eval queries once, so every benchmark variant searches with the same vectors.

    python -m evals.bench.embed_queries            # writes evals/bench/query_vectors.jsonl
    python -m evals.bench.embed_queries --check    # is the file up to date with ground_truth.jsonl?

Why a file instead of embedding during the run:
  - identical input: OpenAI embeddings are not bit-for-bit repeatable, and the
    comparison should differ only in the index, not in the query vectors;
  - clean latency: an API call per search adds 100-300 ms of network time that
    has nothing to do with the database;
  - reproducible: the file is committed, so the benchmark reruns without an
    OpenAI key (one batched call, a fraction of a cent, made once).

Format (JSON Lines, one case per line so diffs stay readable):
    line 1   {"model": ..., "dimensions": ..., "created_at": ...}
    then     {"id": ..., "query": ..., "vector": [...]}

Each vector is stored next to the query text it came from. The benchmark refuses
the file if a case's query changed, a case is missing, or the embedding model or
size in the settings no longer matches (a vector from another model is not
comparable with the catalog's).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.core import llm
from app.core.config import get_settings
from evals.dataset import DEFAULT_PATH, EvalCase, load_cases

VECTORS_PATH = Path(__file__).parent / "query_vectors.jsonl"


class StaleVectorsError(Exception):
    """The vector file doesn't match the eval set or the embedding settings."""


def load_query_vectors(cases: list[EvalCase], path: Path = VECTORS_PATH) -> dict[str, list[float]]:
    """
    Vectors for `cases`, keyed by case id. Raises StaleVectorsError (with the fix)
    if any case has no vector, or its vector was made from different query text,
    another model or another size.
    """
    fix = "run: python -m evals.bench.embed_queries"
    if not path.exists():
        raise StaleVectorsError(f"{path} doesn't exist; {fix}")

    lines = path.read_text(encoding="utf-8").splitlines()
    meta = json.loads(lines[0])
    s = get_settings()
    if (meta["model"], meta["dimensions"]) != (s.embedding_model, s.embedding_dimensions):
        raise StaleVectorsError(
            f"vectors are {meta['model']} ({meta['dimensions']} dims), settings say "
            f"{s.embedding_model} ({s.embedding_dimensions} dims); {fix}"
        )

    stored = {e["id"]: e for e in map(json.loads, lines[1:])}
    problems = []
    for case in cases:
        entry = stored.get(case.id)
        if entry is None:
            problems.append(f"{case.id}: no vector")
        elif entry["query"] != case.query:
            problems.append(f"{case.id}: query changed ({entry['query']!r} -> {case.query!r})")
        elif len(entry["vector"]) != s.embedding_dimensions:
            problems.append(f"{case.id}: {len(entry['vector'])} dims")
    if problems:
        raise StaleVectorsError("; ".join(problems[:5]) + (" ..." if len(problems) > 5 else "") + f"; {fix}")
    return {case.id: stored[case.id]["vector"] for case in cases}


async def embed_cases(cases: list[EvalCase], path: Path = VECTORS_PATH) -> int:
    """Embed every case's query in one batch and write the file. Returns the tokens used."""
    # One request for the unique texts (two cases may share a query).
    texts = sorted({c.query for c in cases})
    result = await llm.embed_texts(texts)
    by_text = dict(zip(texts, result.vectors))

    s = get_settings()
    meta = {
        "model": s.embedding_model,
        "dimensions": s.embedding_dimensions,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cases": len(cases),
    }
    lines = [json.dumps(meta)]
    lines += [json.dumps({"id": c.id, "query": c.query, "vector": by_text[c.query]}, ensure_ascii=False)
              for c in cases]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result.tokens


async def main() -> None:
    ap = argparse.ArgumentParser(description="Embed the eval queries once for the vector-index benchmark.")
    ap.add_argument("--cases", type=Path, default=DEFAULT_PATH, help="eval set (default: evals/ground_truth.jsonl)")
    ap.add_argument("--out", type=Path, default=VECTORS_PATH, help="where to write the vectors")
    ap.add_argument("--check", action="store_true", help="only check that the file is up to date")
    args = ap.parse_args()

    cases = load_cases(args.cases)
    if not cases:
        sys.exit(f"no cases in {args.cases}")

    if args.check:
        try:
            load_query_vectors(cases, args.out)
        except StaleVectorsError as e:
            sys.exit(f"stale: {e}")
        print(f"up to date: {len(cases)} cases in {args.out}")
        return

    tokens = await embed_cases(cases, args.out)
    load_query_vectors(cases, args.out)          # read it back: what the benchmark will see
    s = get_settings()
    print(f"wrote {len(cases)} vectors ({s.embedding_model}, {s.embedding_dimensions} dims, "
          f"{tokens} tokens) to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
