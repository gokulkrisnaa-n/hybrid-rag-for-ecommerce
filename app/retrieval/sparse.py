"""
Lexical search: BM25 over search_text, using the ParadeDB index.

Good at exact words: brand and model names, style codes, colorways.
Weak on meaning: "something warm for winter" shares no words with "fleece".
"""

from __future__ import annotations

from dataclasses import dataclass

from psycopg import AsyncConnection, sql

from app.retrieval.filters import build_where
from app.schemas.query_models import SearchFilters


@dataclass
class Hit:
    product_id: int
    rank: int          # 1-based
    score: float       # BM25 score (higher is better) or cosine distance (lower is better)


# `|||` matches documents containing ANY of the query's words (after the
# index's English stemming), and pdb.score() is the BM25 relevance score.
# The query text is a parameter, so it is matched as words, never parsed as
# query syntax.
_SPARSE_SQL = sql.SQL("""
SELECT product_id, pdb.score(product_id) AS score
FROM clean.products
WHERE search_text ||| %(q)s AND {where}
ORDER BY score DESC, product_id
LIMIT %(k)s
""")


async def search_sparse(conn: AsyncConnection, text: str, filters: SearchFilters, k: int) -> list[Hit]:
    where = build_where(filters)
    query = _SPARSE_SQL.format(where=where.clause)
    cur = await conn.execute(query, {"q": text, "k": k, **where.params})
    rows = await cur.fetchall()
    return [Hit(r["product_id"], i, float(r["score"])) for i, r in enumerate(rows, start=1)]