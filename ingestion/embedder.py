"""
Embed step for the Kith ingestion pipeline.

Reads the embedding queue from clean.products, embeds each product's
search_text with the configured OpenAI model, and writes the vector back.

    clean.products (queue)  ->  app.core.llm.embed_texts()  ->  clean.products.embedding

A product is in the queue when it is active and either
  - its search_text changed since it was embedded (hash differs), or
  - it was embedded with a different model than the one configured now.
So a re-parse only re-embeds products whose text actually changed, and
changing EMBEDDING_MODEL re-embeds everything automatically.

Run from the project root:
    python -m ingestion.embedder --limit 100     # trial run
    python -m ingestion.embedder                 # everything in the queue
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from dataclasses import dataclass, field

import psycopg
from openai import APIConnectionError, APIStatusError, AuthenticationError, BadRequestError

from app.core import llm
from app.core.config import get_settings
from ingestion import db

log = logging.getLogger("embedder")


# ===========================================================================
# SQL
# ===========================================================================

_QUEUE_WHERE = """
    is_active
    AND (embedded_hash IS DISTINCT FROM search_text_hash
         OR embedding_model IS DISTINCT FROM %(model)s)
    AND NOT (product_id = ANY(%(skip)s))
"""

_COUNT_QUEUE_SQL = f"SELECT count(*) AS n FROM clean.products WHERE {_QUEUE_WHERE}"

_FETCH_QUEUE_SQL = f"""
SELECT product_id, search_text, search_text_hash
FROM clean.products
WHERE {_QUEUE_WHERE}
ORDER BY product_id
LIMIT %(limit)s
"""

# The hash in the WHERE clause guards against a re-parse that happened while
# we were embedding: if the text changed, this vector is stale and is not
# written; the product stays in the queue for the next run.
_WRITE_SQL = """
UPDATE clean.products SET
    embedding       = %(embedding)s::vector,
    embedding_model = %(model)s,
    embedded_hash   = %(hash)s,
    embedded_at     = now()
WHERE product_id = %(product_id)s AND search_text_hash = %(hash)s
"""

_COLUMN_DIMS_SQL = """
SELECT atttypmod AS dims
FROM pg_attribute
WHERE attrelid = 'clean.products'::regclass AND attname = 'embedding' AND NOT attisdropped
"""


# ===========================================================================
# Types
# ===========================================================================

@dataclass
class QueuedText:
    product_id: int
    text: str
    text_hash: str


@dataclass
class EmbedStats:
    embedded: int = 0
    stale: int = 0                       # text changed mid-run; picked up next time
    failed: dict[int, str] = field(default_factory=dict)
    tokens: int = 0


# ===========================================================================
# Helpers
# ===========================================================================

def check_dimensions(conn: psycopg.Connection, expected: int) -> None:
    """Fail fast if the schema's vector(N) column doesn't match EMBEDDING_DIMENSIONS."""
    row = conn.execute(_COLUMN_DIMS_SQL).fetchone()
    if row is None:
        raise RuntimeError("clean.products.embedding not found; has db/schema.sql been applied?")
    if row["dims"] != expected:
        raise RuntimeError(
            f"clean.products.embedding is vector({row['dims']}) but EMBEDDING_DIMENSIONS={expected}. "
            "Change one so they match (a column change also needs the HNSW index rebuilt)."
        )


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(map(str, vec)) + "]"


def _short_error(e: Exception) -> str:
    if isinstance(e, APIStatusError):
        return f"HTTP {e.status_code}: {e.message}"[:300]
    return f"{type(e).__name__}: {e}"[:300]


async def _embed_isolating(
    items: list[QueuedText],
) -> tuple[list[tuple[QueuedText, list[float]]], int, list[tuple[QueuedText, str]]]:
    """
    Embed a batch. If OpenAI rejects it (400), split it in half and retry each
    half, so one bad product costs a few extra calls instead of the whole batch.
    Returns (embedded pairs, tokens used, failures).
    Other errors (auth, quota, network after retries) propagate and stop the run.
    """
    try:
        result = await llm.embed_texts([i.text for i in items])
        return list(zip(items, result.vectors)), result.tokens, []
    except BadRequestError as e:
        if len(items) == 1:
            return [], 0, [(items[0], _short_error(e))]
        mid = len(items) // 2
        left, right = await asyncio.gather(
            _embed_isolating(items[:mid]), _embed_isolating(items[mid:])
        )
        return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def _write(conn: psycopg.Connection, pairs: list[tuple[QueuedText, list[float]]], model: str) -> int:
    """Write one page of vectors in a single transaction. Returns rows actually updated."""
    if not pairs:
        return 0
    params = [
        {
            "product_id": item.product_id,
            "embedding": _vector_literal(vec),
            "model": model,
            "hash": item.text_hash,
        }
        for item, vec in pairs
    ]
    written = 0
    with conn.transaction():
        with conn.cursor() as cur:
            for p in params:
                cur.execute(_WRITE_SQL, p)
                written += cur.rowcount
    return written


# ===========================================================================
# The embed step
# ===========================================================================

async def run_embed(
    conn: psycopg.Connection,
    *,
    batch_size: int = 256,
    concurrency: int = 4,
    limit: int | None = None,
) -> EmbedStats:
    """
    Drain the embedding queue.

    Each page holds `batch_size * concurrency` products; its batches are sent
    to OpenAI concurrently, then written in one transaction. Products OpenAI
    rejects are logged and skipped for the rest of this run (they stay in the
    queue). An auth, quota or network error stops the run; everything written
    before it is kept, so rerunning simply continues.
    """
    s = get_settings()
    model = s.embedding_model
    check_dimensions(conn, s.embedding_dimensions)

    stats = EmbedStats()
    skip: list[int] = []
    q = {"model": model, "skip": skip}

    pending = conn.execute(_COUNT_QUEUE_SQL, q).fetchone()["n"]
    target = min(pending, limit) if limit else pending
    log.info(
        "%d products need embedding; embedding %d with %s (%d dims).",
        pending, target, model, s.embedding_dimensions,
    )
    if target == 0:
        return stats

    started = time.monotonic()
    while True:
        done = stats.embedded + stats.stale + len(stats.failed)
        if limit is not None and done >= limit:
            break
        page_size = batch_size * concurrency
        if limit is not None:
            page_size = min(page_size, limit - done)

        rows = conn.execute(_FETCH_QUEUE_SQL, {**q, "limit": page_size}).fetchall()
        if not rows:
            break
        items = [QueuedText(r["product_id"], r["search_text"], r["search_text_hash"]) for r in rows]
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

        results = await asyncio.gather(*(_embed_isolating(b) for b in batches))

        pairs = [p for r in results for p in r[0]]
        written = _write(conn, pairs, model)
        stats.embedded += written
        stats.stale += len(pairs) - written
        stats.tokens += sum(r[1] for r in results)
        for item, error in (f for r in results for f in r[2]):
            stats.failed[item.product_id] = error
            skip.append(item.product_id)
            log.warning("Skipping product %d: %s", item.product_id, error)

        done = stats.embedded + stats.stale + len(stats.failed)
        rate = done / max(time.monotonic() - started, 1e-6)
        log.info(
            "Progress: %d/%d (%.0f%%), %s tokens, ~%.0f products/s.",
            done, target, 100 * done / target, f"{stats.tokens:,}", rate,
        )

    log.info(
        "Done: %d embedded, %d stale, %d failed, %s tokens in %.0fs.",
        stats.embedded, stats.stale, len(stats.failed), f"{stats.tokens:,}",
        time.monotonic() - started,
    )
    return stats


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Embed queued products into clean.products.embedding.")
    ap.add_argument("--batch-size", type=int, default=256, help="products per OpenAI request (max 2048)")
    ap.add_argument("--concurrency", type=int, default=4, help="requests in flight at once")
    ap.add_argument("--limit", type=int, default=None, help="embed at most this many (for trial runs)")
    args = ap.parse_args()

    if not 1 <= args.batch_size <= 2048:
        ap.error("--batch-size must be between 1 and 2048")
    if args.concurrency < 1:
        ap.error("--concurrency must be at least 1")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    with db.connect(get_settings().database_url) as conn:
        try:
            stats = asyncio.run(
                run_embed(conn, batch_size=args.batch_size, concurrency=args.concurrency, limit=args.limit)
            )
        except AuthenticationError:
            log.error("OpenAI rejected the API key. Check OPENAI_API_KEY in .env.")
            raise SystemExit(1)
        except APIConnectionError:
            log.error("Could not reach OpenAI (network error after retries). Rerun to continue.")
            raise SystemExit(1)
        except APIStatusError as e:
            hint = " Add credits in the OpenAI billing page." if "quota" in str(e).lower() else ""
            log.error("OpenAI error, stopping: %s.%s Rerun to continue where it left off.", _short_error(e), hint)
            raise SystemExit(1)

    if stats.failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()