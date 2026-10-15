"""
Lexical search: BM25 over search_text, using the ParadeDB index.

Good at exact words: brand and model names, style codes, colorways.
Weak on meaning: "something warm for winter" shares no words with "fleece".

Before searching, `plan_sparse` checks which query words exist in the catalog
at all. Two things use that:
  - misspelled words (no exact match) are retried with fuzzy matching, so
    "kith hoddie" still finds hoodies;
  - `coverage` tells hybrid search how much of the query BM25 can actually
    see. If only "kith" matched, BM25 is ranking on one generic word, and its
    list should count for less in the fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from psycopg import AsyncConnection, sql

from app.retrieval.filters import build_where
from app.retrieval.query_analysis import content_terms, fuzzy_distance
from app.schemas.query_models import SearchFilters


@dataclass
class Hit:
    product_id: int
    rank: int          # 1-based
    score: float       # BM25 score (higher is better) or cosine distance (lower is better)


@dataclass
class SparsePlan:
    terms: list[str]                                   # content words in the query
    exact: list[str] = field(default_factory=list)     # found as typed (after stemming)
    fuzzy: dict[str, int] = field(default_factory=dict)  # not found as typed, found within N edits
    missing: list[str] = field(default_factory=list)   # not found at all

    @property
    def coverage(self) -> float:
        """Share of the query BM25 can match: exact words count 1, fuzzy ones 0.5."""
        if not self.terms:
            return 1.0
        return (len(self.exact) + 0.5 * len(self.fuzzy)) / len(self.terms)

    def as_debug(self) -> dict:
        return {"coverage": round(self.coverage, 3), "exact": self.exact,
                "fuzzy": self.fuzzy, "missing": self.missing}


# `|||` matches documents containing ANY of the query's words (after the
# index's English stemming), and pdb.score() is the BM25 relevance score.
# The query text is a parameter, so it is matched as words, never parsed as
# query syntax.
_SPARSE_SQL = sql.SQL("""
SELECT product_id, pdb.score(product_id) AS score
FROM clean.products
WHERE ({match}) AND {where}
ORDER BY score DESC, product_id
LIMIT %(k)s
""")


def _exists(param: str, fuzzy: int = 0) -> sql.Composable:
    rhs = sql.Placeholder(param)
    if fuzzy:
        # The distance is part of the type, so it can't be a parameter; it is always 1 or 2.
        rhs = sql.SQL("{}::pdb.fuzzy({})").format(rhs, sql.SQL(str(int(fuzzy))))
    return sql.SQL("EXISTS (SELECT 1 FROM clean.products WHERE is_active AND search_text ||| {})").format(rhs)


async def plan_sparse(conn: AsyncConnection, text: str, *, fuzzy: bool = True) -> SparsePlan:
    """One round trip: does each query word (and, if not, a near-miss of it) occur in the catalog?"""
    plan = SparsePlan(terms=content_terms(text))
    if not plan.terms:
        return plan

    cols: list[sql.Composable] = []
    params: dict[str, str] = {}
    for i, term in enumerate(plan.terms):
        params[f"t{i}"] = term
        cols.append(sql.SQL("{} AS {}").format(_exists(f"t{i}"), sql.Identifier(f"e{i}")))
        d = fuzzy_distance(term) if fuzzy else 0
        if d:
            cols.append(sql.SQL("{} AS {}").format(_exists(f"t{i}", d), sql.Identifier(f"f{i}")))

    cur = await conn.execute(sql.SQL("SELECT {}").format(sql.SQL(", ").join(cols)), params)
    row = await cur.fetchone() or {}

    for i, term in enumerate(plan.terms):
        if row.get(f"e{i}"):
            plan.exact.append(term)
        elif row.get(f"f{i}"):
            plan.fuzzy[term] = fuzzy_distance(term)
        else:
            plan.missing.append(term)
    return plan


async def search_sparse(
    conn: AsyncConnection,
    text: str,
    filters: SearchFilters,
    k: int,
    fuzzy_terms: dict[str, int] | None = None,
) -> list[Hit]:
    """
    BM25 search. `fuzzy_terms` (word -> edit distance) are words with no exact
    match; they are also searched with that many typos allowed.
    """
    where = build_where(filters)
    params: dict = {"q": text, "k": k, **where.params}
    match: list[sql.Composable] = [sql.SQL("search_text ||| %(q)s")]
    for d in (1, 2):
        words = [t for t, dist in (fuzzy_terms or {}).items() if dist == d]
        if words:
            params[f"fz{d}"] = " ".join(words)
            match.append(sql.SQL("search_text ||| %(fz{d})s::pdb.fuzzy({d})".format(d=d)))

    query = _SPARSE_SQL.format(match=sql.SQL(" OR ").join(match), where=where.clause)
    cur = await conn.execute(query, params)
    rows = await cur.fetchall()
    return [Hit(r["product_id"], i, float(r["score"])) for i, r in enumerate(rows, start=1)]