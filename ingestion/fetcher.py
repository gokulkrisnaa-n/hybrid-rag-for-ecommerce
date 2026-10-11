"""
Fetcher for the Kith product ingestion pipeline.

Fetches all products from /products.json and validates
against the RawProductRecord pydantic model

Stores each page in raw.products (see run_fetch); the parser reads from there.

Response handling:
  - 200 with products     -> keep paging
  - 200 with no products  -> end of catalog, stop normally
  - 429 / 5xx / network   -> retry with backoff (honours Retry-After)
  - other 4xx (403, 404)  -> stop and raise FetchError

A failed fetch raises instead of returning a partial list, because the
upsert treats missing products as removed. A partial catalog that looked
complete would mark real products inactive.
"""

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from pydantic import ValidationError

from ingestion import db
from ingestion.models import RawProductRecord

log = logging.getLogger("fetcher")

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 4            # attempts after the first one
BASE_BACKOFF = 2.0         # seconds; doubles each retry
MAX_BACKOFF = 60.0
REQUEST_TIMEOUT = 30.0
HEADERS = {"User-Agent": "kith-rag-portfolio-project (educational; contact via GitHub)"}


class FetchError(Exception):
    """The catalog could not be fetched completely."""


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Read Retry-After, which is either seconds ('30') or an HTTP date."""
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError):
        return None


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter: ~2s, 4s, 8s, 16s ... capped."""
    return min(MAX_BACKOFF, BASE_BACKOFF * 2 ** attempt) * random.uniform(0.8, 1.2)


async def _get_products_page(client: httpx.AsyncClient, url: str) -> list[dict]:
    """Fetch one page, retrying temporary failures. Raises FetchError on permanent ones."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await client.get(url)
        except httpx.TransportError as e:            # timeouts, DNS, connection resets
            reason = f"network error: {type(e).__name__}"
            wait = _backoff(attempt)
        else:
            if response.status_code == 200:
                return response.json().get("products", [])

            if response.status_code not in RETRYABLE_STATUS:
                raise FetchError(f"{url} -> HTTP {response.status_code} (not retryable)")

            reason = f"HTTP {response.status_code}"
            wait = _retry_after_seconds(response) or _backoff(attempt)

        if attempt == MAX_RETRIES:
            raise FetchError(f"{url} -> {reason}, gave up after {MAX_RETRIES + 1} attempts")

        log.warning("%s -> %s, retry %d/%d in %.1fs", url, reason, attempt + 1, MAX_RETRIES, wait)
        await asyncio.sleep(wait)

    raise FetchError(f"{url} -> retries exhausted")   # unreachable, keeps type checkers happy


def _to_records(base_url: str, products: list[dict]) -> list[RawProductRecord]:
    """Wrap one page of Shopify products. A malformed product is logged and skipped, not fatal."""
    scraped_at = datetime.now(timezone.utc)
    records = []
    for product in products:
        try:
            records.append(
                RawProductRecord(
                    product_id=str(product.get("id", "")),
                    scraped_at=scraped_at,
                    source_url=f"{base_url}/products/{product.get('handle', '')}",
                    raw=product,
                )
            )
        except ValidationError as e:
            log.warning("Skipping malformed product id=%r: %s", product.get("id"), e.errors()[0]["msg"])
    return records


async def iter_catalog_pages(
    base_url: str,
    max_pages: int = 3,
    *,
    delay_range: tuple[float, float] = (1.0, 5.0),
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[tuple[int, list[RawProductRecord]]]:
    """
    Yield (page_number, records) one page at a time.

    When the catalog ends, the last item yielded is (page_number, []): an
    explicit end-of-catalog marker. If iteration stops without it, the
    fetch stopped at max_pages and the catalog may be incomplete.
    Raises FetchError if a page can't be fetched.
    """
    async with httpx.AsyncClient(
        headers=HEADERS, timeout=REQUEST_TIMEOUT, transport=transport
    ) as client:
        for page in range(1, max_pages + 1):
            hit_url = f"{base_url}/products.json?limit=250&page={page}"
            log.info("Fetching: %s", hit_url)

            products = await _get_products_page(client, hit_url)

            # An empty page means we have reached the end of the catalog.
            if not products:
                log.info("Page %d is empty: end of catalog.", page)
                yield page, []
                return

            yield page, _to_records(base_url, products)

            # Random delay before next hit, so we do not self-DDoS the server
            await asyncio.sleep(random.uniform(*delay_range))

    log.warning("Stopped at max_pages=%d; the catalog may have more pages.", max_pages)


async def fetch_shopify_catalog(base_url: str, max_pages: int = 3, **kwargs) -> list[RawProductRecord]:
    """Fetch the catalog into memory (no database). Handy for quick experiments."""
    catalog: list[RawProductRecord] = []
    async for _, records in iter_catalog_pages(base_url, max_pages, **kwargs):
        catalog.extend(records)
    log.info("Fetched %d products.", len(catalog))
    return catalog


async def run_fetch(conn, base_url: str, max_pages: int = 200, **kwargs) -> dict:
    """
    The fetch step: stream every page into raw.products and record the run.

    Run status:
      complete : reached the empty page; the parser may soft-delete products missing from it
      partial  : stopped at max_pages; never used for soft-deletes
      failed   : FetchError (or any crash); never used for soft-deletes

    Each page is committed as it arrives, so a failure keeps the pages already stored.
    """
    run_id = db.start_fetch_run(conn, base_url)
    totals = db.UpsertCounts()
    reached_end = False

    try:
        # aclosing() shuts the HTTP client down promptly when we break out early.
        async with aclosing(iter_catalog_pages(base_url, max_pages, **kwargs)) as pages:
            async for page, records in pages:
                if not records:
                    reached_end = True
                    break
                counts = db.upsert_raw_products(conn, run_id, records)
                totals.new += counts.new
                totals.changed += counts.changed
                totals.unchanged += counts.unchanged
                log.info(
                    "Page %d stored: %d new, %d changed, %d unchanged.",
                    page, counts.new, counts.changed, counts.unchanged,
                )
    except BaseException as e:            # includes Ctrl+C, so the run is never left 'running'
        db.finish_fetch_run(conn, run_id, "failed", error=f"{type(e).__name__}: {e}")
        log.error("Fetch run %d failed: %s", run_id, e)
        raise

    status = "complete" if reached_end else "partial"
    db.finish_fetch_run(conn, run_id, status)
    log.info(
        "Fetch run %d %s: %d products (%d new, %d changed, %d unchanged).",
        run_id, status, totals.seen, totals.new, totals.changed, totals.unchanged,
    )
    return {"run_id": run_id, "status": status, "new": totals.new,
            "changed": totals.changed, "unchanged": totals.unchanged}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fetch the Shopify catalog into raw.products.")
    ap.add_argument("--base-url", default="https://kith.com")
    ap.add_argument("--max-pages", type=int, default=200,
                    help="safety cap; set high enough to reach the end of the catalog")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    with db.connect() as conn:
        asyncio.run(run_fetch(conn, args.base_url, args.max_pages))

