"""
Vector search through ParadeDB's native vector index (0.27+), for the benchmark.

Same contract as app/retrieval/dense.search_dense (query vector, filters, k ->
ranked hits), so hybrid search can run on either. The SQL differs in three ways:

  - `product_id @@@ pdb.all()` hands the query to ParadeDB's custom scan;
  - `ORDER BY embedding <=> q, product_id LIMIT k` is the shape its Top K scan
    accelerates (the operator must match the index's vector_cosine_ops);
  - no MATERIALIZED re-sort: that exists for pgvector's relaxed-order iterative
    scans, and ParadeDB returns rows in distance order.

The filters are the app's own (app/retrieval/filters.build_where), unchanged:
`bench_setup.sh explain` showed plain SQL predicates on indexed columns
(is_active, in_stock, price_min <= x) pushed into the same index scan as Tantivy
terms and ranges. Whether EVERY filter we use is pushed down is what
`probe_pushdown` checks. A filter that isn't shows up in the plan as a Postgres
"Filter:" line, i.e. applied after the index already picked its top k.

`embedding IS NOT NULL` is left out (pgvector's query has it): every product in
the catalog has an embedding, and the condition would be one more thing to push down.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from psycopg import AsyncConnection, sql

from app.retrieval.filters import build_where
from app.retrieval.sparse import Hit
from app.schemas.query_models import SearchFilters

_PDB_DENSE_SQL = sql.SQL("""
SELECT product_id
FROM clean.products
WHERE product_id @@@ pdb.all() AND {where}
ORDER BY embedding <=> %(qvec)s::vector, product_id
LIMIT %(k)s
""")


def vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(map(str, vec)) + "]"


def dense_query(query_vec: list[float], filters: SearchFilters, k: int) -> tuple[sql.Composed, dict]:
    """The SQL and parameters of one ParadeDB vector search."""
    where = build_where(filters)
    return (_PDB_DENSE_SQL.format(where=where.clause),
            {"qvec": vector_literal(query_vec), "k": k, **where.params})


async def search_dense_paradedb(
    conn: AsyncConnection,
    query_vec: list[float],
    filters: SearchFilters,
    k: int,
    *,
    max_probe: int | None = None,
) -> list[Hit]:
    """
    Nearest products by cosine distance, best first. `max_probe` sets
    paradedb.vector_cluster_max_probe for this query only (None = server default).

    The distance itself isn't selected, so the index does all the work; Hit.score
    is 0.0 (hybrid search fuses by rank and never reads it).
    """
    query, params = dense_query(query_vec, filters, k)
    if max_probe is None:
        cur = await conn.execute(query, params)
        rows = await cur.fetchall()
    else:
        async with conn.transaction():
            await conn.execute("SELECT set_config('paradedb.vector_cluster_max_probe', %s, true)",
                               (str(max_probe),))
            cur = await conn.execute(query, params)
            rows = await cur.fetchall()
    return [Hit(r["product_id"], i, 0.0) for i, r in enumerate(rows, start=1)]


# ---------------------------------------------------------------------------
# Is each filter evaluated inside the index?
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    lines: list[str]

    @property
    def top_k(self) -> bool:
        return any("TopKScanExecState" in line for line in self.lines)

    @property
    def post_filter(self) -> str | None:
        """A predicate Postgres applies after the scan ("Filter: ..."), if any."""
        return next((line.strip() for line in self.lines if line.strip().startswith("Filter:")), None)

    @property
    def tantivy_query(self) -> str | None:
        line = next((line.strip() for line in self.lines if line.strip().startswith("Tantivy Query:")), None)
        return line.removeprefix("Tantivy Query:").strip() if line else None

    @property
    def verdict(self) -> str:
        if not self.top_k:
            return "no Top K scan (Postgres sorts)"
        if self.post_filter:
            return "Top K, then filtered by Postgres"
        return "in the index"

    def text(self) -> str:
        return "\n".join(self.lines)


async def explain(conn: AsyncConnection, query: sql.Composable, params: dict) -> Plan:
    cur = await conn.execute(sql.SQL("EXPLAIN (COSTS OFF) {}").format(query), params)
    rows = await cur.fetchall()
    return Plan([next(iter(r.values())) if isinstance(r, dict) else r[0] for r in rows])


def probe_filters(subcategory: str | None = None) -> dict[str, SearchFilters]:
    """One filter at a time, each written by build_where exactly as search writes it."""
    shapes = {
        "is_active, in_stock (default)": SearchFilters(),
        "sold out included": SearchFilters(in_stock_only=False),
        "on_sale": SearchFilters(on_sale_only=True),
        "price_min <= x (max price)": SearchFilters(max_price=Decimal(100)),
        "price_max >= x (min price)": SearchFilters(min_price=Decimal(100)),
        "gender = ANY(...)": SearchFilters(genders=["womens", "unisex"]),
        "NOT gender = ANY(...) (exclude)": SearchFilters(exclude_genders=["kids"]),
        "category = ANY(...)": SearchFilters(categories=["footwear"]),
        "lower(brand) = ANY(...) OR collaborator": SearchFilters(brands=["nike"]),
        "NOT lower(coalesce(brand)) ... (exclude)": SearchFilters(exclude_brands=["nike"]),
        "sizes_available && ... (not indexed)": SearchFilters(sizes=["10"]),
    }
    if subcategory:
        shapes["subcategory = ANY(...)"] = SearchFilters(subcategories=[subcategory])
    return shapes


@dataclass
class Pushdown:
    filter: str
    plan: Plan


async def probe_pushdown(conn: AsyncConnection, query_vec: list[float], k: int) -> list[Pushdown]:
    """EXPLAIN the vector search under each filter in probe_filters()."""
    cur = await conn.execute(
        "SELECT subcategory FROM clean.products WHERE subcategory IS NOT NULL "
        "GROUP BY subcategory ORDER BY count(*) DESC LIMIT 1")
    row = await cur.fetchone()
    out = []
    for name, filters in probe_filters(row["subcategory"] if row else None).items():
        query, params = dense_query(query_vec, filters, k)
        out.append(Pushdown(name, await explain(conn, query, params)))
    return out
