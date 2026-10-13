"""
Fetcher for the Kith product ingestion pipeline.

Fetches all products from /products.json and validates
against the RawProductRecord pydantic model

Stores each page in raw.products (see run_fetch); the parser reads from there.

Response handling:
  - 200 with products     -> keep paging
  - 200 with no products  -> end of catalog, stop normally
  - 429 / 5xx / network   -> retry with backoff (honours Retry-After)
  - 400 after a full page -> Shopify's pagination cap; stop, run is partial
  - other 4xx (403, 404)  -> stop and raise FetchError

Shopify stops serving /products.json pages after 25,000 products (100 pages
of 250); asking for page 101 returns HTTP 400. The fetcher stops at the cap
without requesting that page, and also treats a 400 straight after a full
page as the cap, in case the limit isn't exactly 25,000.

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

# Shopify stops paginating storefront product lists after 25,000 items
# (page 101 of 250 returns HTTP 400). A crawl that reaches the cap is recorded
# as partial, never complete, so soft-deletes are never computed from a
# truncated catalog.
SHOPIFY_PAGINATION_CAP = 25_000
PAGE_SIZE = 250


class FetchError(Exception):
    """The catalog could not be fetched completely."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code     # HTTP status, when the failure was an HTTP response


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
                raise FetchError(
                    f"{url} -> HTTP {response.status_code} (not retryable)",
                    status_code=response.status_code,
                )

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
    state: dict | None = None,
) -> AsyncIterator[tuple[int, list[RawProductRecord]]]:
    """
    Yield (page_number, records) one page at a time.

    When the catalog genuinely ends, the last item yielded is (page_number, []):
    an explicit end-of-catalog marker. If iteration stops without it, the crawl
    was cut short and the catalog may be incomplete.

    If `state` is given, state["stop_reason"] is set to one of:
      "end"       : reached an empty page below the Shopify cap
      "cap"       : hit Shopify's pagination cap (reached SHOPIFY_PAGINATION_CAP
                    products, or got HTTP 400 right after a full page); more may exist
      "max_pages" : stopped at max_pages
    state["products"] is the number of products returned.
    Raises FetchError if a page can't be fetched for any other reason.
    """
    state = state if state is not None else {}
    products_returned = 0
    last_page_full = False

    async with httpx.AsyncClient(
        headers=HEADERS, timeout=REQUEST_TIMEOUT, transport=transport
    ) as client:
        for page in range(1, max_pages + 1):
            hit_url = f"{base_url}/products.json?limit={PAGE_SIZE}&page={page}"
            log.info("Fetching: %s", hit_url)

            try:
                products = await _get_products_page(client, hit_url)
            except FetchError as e:
                # A 400 right after a full page is Shopify refusing to paginate further,
                # not a broken request. A 400 on page 1 or after a short page is a real error.
                if e.status_code == 400 and last_page_full:
                    state.update(stop_reason="cap", products=products_returned)
                    log.warning(
                        "Page %d returned HTTP 400 after a full page: Shopify's pagination cap "
                        "(%d products fetched). The catalog may be larger; marking this run partial.",
                        page, products_returned,
                    )
                    return
                raise

            # An empty page means the end of the catalog.
            if not products:
                state.update(stop_reason="end", products=products_returned)
                log.info("Page %d is empty: end of catalog.", page)
                yield page, []
                return

            products_returned += len(products)
            last_page_full = len(products) >= PAGE_SIZE
            yield page, _to_records(base_url, products)

            # The next page would be refused (HTTP 400), so don't ask for it.
            if products_returned >= SHOPIFY_PAGINATION_CAP and last_page_full:
                state.update(stop_reason="cap", products=products_returned)
                log.warning(
                    "Reached Shopify's %d-product pagination cap; the catalog may be larger. "
                    "Marking this run partial.", SHOPIFY_PAGINATION_CAP,
                )
                return

            # Random delay before next hit, so we do not self-DDoS the server
            await asyncio.sleep(random.uniform(*delay_range))

    state.update(stop_reason="max_pages", products=products_returned)
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
      partial  : stopped at max_pages or at Shopify's 25,000-product cap;
                 never used for soft-deletes
      failed   : FetchError (or any crash); never used for soft-deletes

    Each page is committed as it arrives, so a failure keeps the pages already stored.
    """
    run_id = db.start_fetch_run(conn, base_url)
    totals = db.UpsertCounts()
    reached_end = False
    state: dict = {}

    try:
        # aclosing() shuts the HTTP client down promptly when we break out early.
        async with aclosing(iter_catalog_pages(base_url, max_pages, state=state, **kwargs)) as pages:
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
    note = {
        "cap": f"stopped at Shopify's pagination cap after {state.get('products', 0)} products",
        "max_pages": f"stopped at max_pages={max_pages}",
    }.get(state.get("stop_reason"))
    db.finish_fetch_run(conn, run_id, status, error=note)
    log.info(
        "Fetch run %d %s: %d products (%d new, %d changed, %d unchanged).",
        run_id, status, totals.seen, totals.new, totals.changed, totals.unchanged,
    )
    return {"run_id": run_id, "status": status, "stop_reason": state.get("stop_reason"),
            "new": totals.new, "changed": totals.changed, "unchanged": totals.unchanged}


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
