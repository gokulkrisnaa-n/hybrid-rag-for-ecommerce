"""
Database access for the Kith ingestion pipeline (psycopg 3).

All SQL lives here, so fetcher.py and parser.py never write SQL themselves.

Fetcher side
    start_fetch_run()        -> new row in raw.fetch_runs (status 'running')
    upsert_raw_products()    -> insert/refresh raw.products for one page
    finish_fetch_run()       -> mark the run complete / partial / failed

Parser side
    known_vendors()          -> every vendor in raw.products (brand vocabulary)
    fetch_parse_queue()      -> raw rows whose payload hasn't been parsed yet
    save_clean_product()     -> upsert clean.products + replace its variants
    save_parse_failure()     -> record why a product couldn't be parsed
    sync_active_flags()      -> soft-delete / restore based on the latest COMPLETE run

Maintenance
    requeue_failed()         -> retry failed products (after fixing the parser)
    requeue_all()            -> re-parse everything (after changing parsing rules)

Connection: DATABASE_URL, e.g.
    postgresql://kith:password@localhost:5433/kith_rag
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ingestion.models import CleanProduct, RawProductRecord

from dotenv import load_dotenv
load_dotenv()


# ===========================================================================
# Connection and hashing
# ===========================================================================

def connect(url: Optional[str] = None) -> psycopg.Connection:
    """
    Open a connection. Rows come back as dicts.

    autocommit=True is deliberate: single statements commit on their own, and
    every `with conn.transaction():` block below is a real transaction that
    commits or rolls back as a unit. (Without autocommit, those blocks become
    savepoints inside an implicit transaction and nothing commits until later.)
    """
    url = url or os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("Set DATABASE_URL (see .env.example).")
    return psycopg.connect(url, row_factory=dict_row, autocommit=True)


def payload_hash(payload: dict[str, Any]) -> str:
    """
    Stable sha256 of a product's JSON. Keys are sorted so the same product
    always hashes the same, whatever order Shopify sends the fields in.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ===========================================================================
# Fetcher side
# ===========================================================================

@dataclass
class UpsertCounts:
    new: int = 0
    changed: int = 0
    unchanged: int = 0

    @property
    def seen(self) -> int:
        return self.new + self.changed + self.unchanged


def start_fetch_run(conn: psycopg.Connection, source: str) -> int:
    """Create a fetch run in status 'running'. It commits immediately, so it's visible even if we crash."""
    run_id = conn.execute(
        "INSERT INTO raw.fetch_runs (source) VALUES (%s) RETURNING run_id", (source,)
    ).fetchone()["run_id"]
    return run_id


# Every product seen gets last_seen_* refreshed. Payload and last_changed_*
# are only rewritten when the hash differs, so an unchanged product costs
# almost nothing and never re-enters the parser queue.
_UPSERT_RAW_SQL = """
INSERT INTO raw.products AS p (
    product_id, handle, payload, payload_hash, shopify_updated_at,
    first_seen_run, last_seen_run, last_changed_run,
    first_seen_at, last_seen_at, last_changed_at
)
VALUES (%(product_id)s, %(handle)s, %(payload)s, %(payload_hash)s, %(shopify_updated_at)s,
        %(run_id)s, %(run_id)s, %(run_id)s, %(seen_at)s, %(seen_at)s, %(seen_at)s)
ON CONFLICT (product_id) DO UPDATE SET
    last_seen_run = EXCLUDED.last_seen_run,
    last_seen_at  = EXCLUDED.last_seen_at,
    handle             = CASE WHEN p.payload_hash <> EXCLUDED.payload_hash THEN EXCLUDED.handle             ELSE p.handle             END,
    payload            = CASE WHEN p.payload_hash <> EXCLUDED.payload_hash THEN EXCLUDED.payload            ELSE p.payload            END,
    shopify_updated_at = CASE WHEN p.payload_hash <> EXCLUDED.payload_hash THEN EXCLUDED.shopify_updated_at ELSE p.shopify_updated_at END,
    last_changed_run   = CASE WHEN p.payload_hash <> EXCLUDED.payload_hash THEN EXCLUDED.last_changed_run   ELSE p.last_changed_run   END,
    last_changed_at    = CASE WHEN p.payload_hash <> EXCLUDED.payload_hash THEN EXCLUDED.last_changed_at    ELSE p.last_changed_at    END,
    payload_hash  = EXCLUDED.payload_hash
RETURNING
    (xmax = 0)                                          AS inserted,
    (last_changed_run = %(run_id)s AND xmax <> 0)       AS changed
"""


def upsert_raw_products(
    conn: psycopg.Connection, run_id: int, records: list[RawProductRecord], *, page_fetched: bool = True
) -> UpsertCounts:
    """
    Upsert one page of products and update the run's counters, in one transaction.
    Returns how many were new, changed or unchanged.
    """
    # Shopify pagination can repeat a product within a page; one statement
    # can't update the same row twice, so keep the last occurrence.
    unique = {r.product_id: r for r in records}

    rows = [
        {
            "product_id": int(r.product_id),
            "handle": str(r.raw.get("handle", "")),
            "payload": Jsonb(r.raw),
            "payload_hash": payload_hash(r.raw),
            "shopify_updated_at": r.raw.get("updated_at"),
            "run_id": run_id,
            "seen_at": r.scraped_at,
        }
        for r in unique.values()
    ]

    counts = UpsertCounts()
    with conn.transaction():
        with conn.cursor() as cur:
            if rows:
                cur.executemany(_UPSERT_RAW_SQL, rows, returning=True)
                while True:
                    res = cur.fetchone()
                    if res["inserted"]:
                        counts.new += 1
                    elif res["changed"]:
                        counts.changed += 1
                    else:
                        counts.unchanged += 1
                    if not cur.nextset():
                        break

            cur.execute(
                """
                UPDATE raw.fetch_runs SET
                    pages_fetched    = pages_fetched + %s,
                    products_seen    = products_seen + %s,
                    products_new     = products_new + %s,
                    products_changed = products_changed + %s
                WHERE run_id = %s
                """,
                (1 if page_fetched else 0, counts.seen, counts.new, counts.changed, run_id),
            )
    return counts


def finish_fetch_run(
    conn: psycopg.Connection, run_id: int, status: str, error: Optional[str] = None
) -> None:
    """status: 'complete' (reached the end), 'partial' (stopped at max_pages) or 'failed'."""
    if status not in {"complete", "partial", "failed"}:
        raise ValueError(f"bad status {status!r}")
    conn.execute(
        "UPDATE raw.fetch_runs SET status = %s, finished_at = now(), error = %s WHERE run_id = %s",
        (status, error, run_id),
    )


# ===========================================================================
# Parser side
# ===========================================================================

@dataclass
class QueuedProduct:
    record: RawProductRecord
    payload_hash: str           # the exact payload version being parsed


def known_vendors(conn: psycopg.Connection) -> set[str]:
    """Every vendor in the raw catalog, used as the brand vocabulary for collab parsing."""
    rows = conn.execute(
        "SELECT DISTINCT payload->>'vendor' AS vendor FROM raw.products "
        "WHERE coalesce(payload->>'vendor', '') <> ''"
    ).fetchall()
    return {r["vendor"].strip() for r in rows}


def fetch_parse_queue(conn: psycopg.Connection, limit: int = 500) -> list[QueuedProduct]:
    """Raw products whose current payload hasn't been parsed yet (new, changed or requeued)."""
    rows = conn.execute(
        """
        SELECT p.product_id, p.handle, p.payload, p.payload_hash, p.last_seen_at, r.source
        FROM raw.products p
        JOIN raw.fetch_runs r ON r.run_id = p.last_seen_run
        WHERE p.parsed_hash IS DISTINCT FROM p.payload_hash
        ORDER BY p.product_id
        LIMIT %s
        """,
        (limit,),
    ).fetchall()

    return [
        QueuedProduct(
            record=RawProductRecord(
                product_id=str(r["product_id"]),
                scraped_at=r["last_seen_at"],
                source_url=f"{r['source'].rstrip('/')}/products/{r['handle']}",
                raw=r["payload"],
            ),
            payload_hash=r["payload_hash"],
        )
        for r in rows
    ]


# Columns the parser owns. Embedding columns and soft-delete state are
# deliberately absent, so a re-parse never wipes an embedding.
_CLEAN_COLUMNS = [
    "product_id", "handle", "url", "image_url",
    "title", "brand", "collaborator", "collaborator_note", "model", "colorway",
    "style_code", "product_type", "category", "subcategory", "gender",
    "materials", "description_bullets", "clean_tags", "purchase_limit",
    "price_min", "price_max", "compare_at_price", "on_sale", "in_stock", "sizes_available",
    "search_text", "search_text_hash",
    "shopify_created_at", "shopify_updated_at", "shopify_published_at", "scraped_at",
]
_UPSERT_CLEAN_SQL = (
    f"INSERT INTO clean.products ({', '.join(_CLEAN_COLUMNS)}, parsed_at) "
    f"VALUES ({', '.join('%(' + c + ')s' for c in _CLEAN_COLUMNS)}, now()) "
    "ON CONFLICT (product_id) DO UPDATE SET "
    + ", ".join(f"{c} = EXCLUDED.{c}" for c in _CLEAN_COLUMNS if c != "product_id")
    + ", parsed_at = now()"
)

_VARIANT_COLUMNS = ["variant_id", "product_id", "size", "sku", "price", "compare_at_price", "on_sale", "available"]
_INSERT_VARIANT_SQL = (
    f"INSERT INTO clean.variants ({', '.join(_VARIANT_COLUMNS)}) "
    f"VALUES ({', '.join('%(' + c + ')s' for c in _VARIANT_COLUMNS)})"
)

# Only mark parsed if the payload is still the version we parsed; if the
# fetcher replaced it meanwhile, the row stays in the queue.
_MARK_PARSED_SQL = """
UPDATE raw.products
SET parsed_hash = payload_hash, parsed_at = now(), parse_status = %s
WHERE product_id = %s AND payload_hash = %s
"""


def save_clean_product(conn: psycopg.Connection, product: CleanProduct, parsed_hash: str) -> None:
    """
    Upsert the product, replace its variants, clear any old failure and mark
    the raw row parsed, all in one transaction.
    """
    row = product.product_row()
    row["product_id"] = int(row["product_id"])
    row["url"] = str(row["url"])
    row["image_url"] = str(row["image_url"]) if row["image_url"] else None

    variants = [
        {**v, "product_id": int(v["product_id"]), "variant_id": int(v["variant_id"])}
        for v in product.variant_rows()
    ]

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(_UPSERT_CLEAN_SQL, row)
            cur.execute("DELETE FROM clean.variants WHERE product_id = %s", (row["product_id"],))
            cur.executemany(_INSERT_VARIANT_SQL, variants)
            cur.execute("DELETE FROM clean.parse_failures WHERE product_id = %s", (row["product_id"],))
            cur.execute(_MARK_PARSED_SQL, ("ok", row["product_id"], parsed_hash))


def save_parse_failure(
    conn: psycopg.Connection, product_id: str, parsed_hash: str, title: Optional[str], error: str
) -> None:
    """
    Record the failure and mark this payload version as processed, so it isn't
    retried on every run. It re-enters the queue when the payload changes, or
    when you call requeue_failed() after fixing the parser.
    The product's previous clean row, if any, is left as it was.
    """
    pid = int(product_id)
    with conn.transaction():
        conn.execute(
            """
            INSERT INTO clean.parse_failures (product_id, payload_hash, title, error, failed_at)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (product_id) DO UPDATE SET
                payload_hash = EXCLUDED.payload_hash, title = EXCLUDED.title,
                error = EXCLUDED.error, failed_at = EXCLUDED.failed_at
            """,
            (pid, parsed_hash, title, error[:2000]),
        )
        conn.execute(_MARK_PARSED_SQL, ("failed", pid, parsed_hash))


@dataclass
class ActiveSync:
    run_id: Optional[int]
    deactivated: int = 0
    reactivated: int = 0


def sync_active_flags(conn: psycopg.Connection, source: str) -> ActiveSync:
    """
    Soft-delete products missing from the latest COMPLETE fetch run, and
    restore any that reappeared. Partial and failed runs are never used, so a
    rate-limited fetch can't wipe the catalog.
    """
    latest = conn.execute(
        "SELECT max(run_id) AS run_id FROM raw.fetch_runs WHERE source = %s AND status = 'complete'",
        (source,),
    ).fetchone()["run_id"]
    if latest is None:
        return ActiveSync(run_id=None)

    with conn.transaction():
        deactivated = conn.execute(
            """
            UPDATE clean.products c SET is_active = false, deactivated_at = now()
            FROM raw.products p
            WHERE p.product_id = c.product_id AND c.is_active AND p.last_seen_run < %s
            """,
            (latest,),
        ).rowcount
        reactivated = conn.execute(
            """
            UPDATE clean.products c SET is_active = true, deactivated_at = NULL
            FROM raw.products p
            WHERE p.product_id = c.product_id AND NOT c.is_active AND p.last_seen_run = %s
            """,
            (latest,),
        ).rowcount
    return ActiveSync(run_id=latest, deactivated=deactivated, reactivated=reactivated)


# ===========================================================================
# Maintenance
# ===========================================================================

def requeue_failed(conn: psycopg.Connection) -> int:
    """Put every failed product back in the parse queue. Use after fixing a parser bug."""
    with conn.transaction():
        return conn.execute(
            "UPDATE raw.products SET parsed_hash = NULL, parsed_at = NULL, parse_status = NULL "
            "WHERE parse_status = 'failed'"
        ).rowcount


def requeue_all(conn: psycopg.Connection) -> int:
    """
    Put every product back in the parse queue. Use after changing parsing rules.
    Embeddings are kept; only products whose search text actually changes
    will be re-embedded.
    """
    with conn.transaction():
        return conn.execute(
            "UPDATE raw.products SET parsed_hash = NULL, parsed_at = NULL, parse_status = NULL"
        ).rowcount
    