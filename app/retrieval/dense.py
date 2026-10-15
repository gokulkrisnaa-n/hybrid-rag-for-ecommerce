"""
Semantic search: cosine distance between the query embedding and product
embeddings, using the HNSW index.

Good at meaning: "something warm for winter" finds fleeces and parkas.
Weak on exact tokens: a style code like "aaki4323" means little to an embedding.
"""

from __future__ import annotations

from psycopg import AsyncConnection, sql

from app.core.config import get_settings
from app.retrieval.filters import build_where
from app.retrieval.sparse import Hit
from app.schemas.query_models import SearchFilters

# The MATERIALIZED CTE re-sorts by exact distance, because iterative scans in
# relaxed_order can return rows slightly out of order. "+ 0" stops Postgres
# 17+ from optimising that re-sort away (pgvector docs).
_DENSE_SQL = sql.SQL("""
WITH candidates AS MATERIALIZED (
    SELECT product_id, embedding <=> %(qvec)s::vector AS distance
    FROM clean.products
    WHERE embedding IS NOT NULL AND {where}
    ORDER BY distance
    LIMIT %(k)s
)
SELECT product_id, distance FROM candidates ORDER BY distance + 0, product_id
""")

_iterative_supported: bool | None = None


async def _supports_iterative_scan(conn: AsyncConnection) -> bool:
    """Iterative index scans arrived in pgvector 0.8.0. Checked once per process."""
    global _iterative_supported
    if _iterative_supported is None:
        cur = await conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = await cur.fetchone()
        try:
            version = tuple(int(x) for x in row["extversion"].split(".")[:2]) if row else (0, 0)
        except ValueError:
            version = (0, 0)
        _iterative_supported = version >= (0, 8)
    return _iterative_supported


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(map(str, vec)) + "]"


async def search_dense(conn: AsyncConnection, query_vec: list[float], filters: SearchFilters, k: int) -> list[Hit]:
    s = get_settings()
    where = build_where(filters)
    query = _DENSE_SQL.format(where=where.clause)

    # Settings apply to this transaction only (set_config(..., true)), so a
    # pooled connection never carries them into someone else's query.
    async with conn.transaction():
        await conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(max(s.hnsw_ef_search, k)),))
        if s.hnsw_iterative_scan != "off" and await _supports_iterative_scan(conn):
            await conn.execute("SELECT set_config('hnsw.iterative_scan', %s, true)", (s.hnsw_iterative_scan,))
        cur = await conn.execute(query, {"qvec": _vector_literal(query_vec), "k": k, **where.params})
        rows = await cur.fetchall()

    return [Hit(r["product_id"], i, float(r["distance"])) for i, r in enumerate(rows, start=1)]