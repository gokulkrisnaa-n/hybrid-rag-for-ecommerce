"""
Round 1 retrieval rules: code routing, fuzzy fallback, coverage weighting.

No database or OpenAI needed: the search functions are replaced with fakes,
so these tests check the decisions, not ParadeDB itself.

    pytest tests/test_retrieval_rules.py -q
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from decimal import Decimal

import pytest

from app.retrieval import hybrid, sparse
from app.retrieval.query_analysis import content_terms, fuzzy_distance, is_code_query
from app.retrieval.sparse import Hit, SparsePlan
from app.schemas.product_models import ProductCard
from app.schemas.query_models import SearchFilters, SearchQuery


# ---------------------------------------------------------------------------
# query_analysis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", ["aaki4323", "ai1203a899-105", "CH6101-32-SS26", " khma030255-001 "])
def test_style_codes_are_codes(q):
    assert is_code_query(q)


@pytest.mark.parametrize("q", ["990v6", "1906r", "chuck 70", "samba", "gel kayano 12.1", "12345678", "", "a$ap"])
def test_names_and_words_are_not_codes(q):
    assert not is_code_query(q)


def test_content_terms_drop_stopwords_short_tokens_and_duplicates():
    assert content_terms("Looking for a cozy hoodie for my girlfriend") == ["looking", "cozy", "hoodie", "my", "girlfriend"]
    assert content_terms("women's winter parka") == ["women", "winter", "parka"]
    assert content_terms("the a for") == []


def test_fuzzy_distance_by_length():
    assert [fuzzy_distance(t) for t in ("tee", "sambaa", "hoddie", "nupste", "kith")] == [0, 2, 2, 2, 1]


def test_coverage():
    assert SparsePlan(terms=[]).coverage == 1.0
    assert SparsePlan(terms=["kith", "hoddie"], exact=["kith"], fuzzy={"hoddie": 2}).coverage == 0.75
    assert SparsePlan(terms=["kith", "hoddie"], exact=["kith"], missing=["hoddie"]).coverage == 0.5


# ---------------------------------------------------------------------------
# plan_sparse and search_sparse build the right SQL
# ---------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, row=None, rows=None):
        self._row, self._rows = row, rows or []

    async def fetchone(self):
        return self._row

    async def fetchall(self):
        return self._rows


class RecordingConn:
    def __init__(self, row=None, rows=None):
        self.row, self.rows, self.calls = row, rows, []

    async def execute(self, query, params=None):
        self.calls.append((query, params))
        return FakeCursor(self.row, self.rows)


def render(query) -> str:
    """Composed SQL as text, without a database connection (placeholders stay %(name)s)."""
    from psycopg import sql

    def walk(q):
        if isinstance(q, sql.Composed):
            return "".join(walk(p) for p in q)
        if isinstance(q, sql.SQL):
            return q._obj
        if isinstance(q, sql.Placeholder):
            return f"%({q._obj})s"
        if isinstance(q, sql.Identifier):
            return '"' + ".".join(q._obj) + '"'
        raise TypeError(type(q))
    return walk(query)


async def test_plan_sparse_classifies_terms():
    conn = RecordingConn(row={"e0": True, "e1": False, "f1": True, "e2": False, "f2": False})
    plan = await sparse.plan_sparse(conn, "kith hoddie zzzzqx", SearchFilters(max_price=100))
    assert plan.exact == ["kith"] and plan.fuzzy == {"hoddie": 2} and plan.missing == ["zzzzqx"]
    text = render(conn.calls[0][0])
    assert "search_text ||| %(t1)s::pdb.fuzzy(2) AND is_active AND in_stock AND price_min <= %(f_max_price)s" in text
    assert text.count("EXISTS") == 6          # exact + fuzzy for each of the three words
    assert conn.calls[0][1] == {"t0": "kith", "t1": "hoddie", "t2": "zzzzqx", "f_max_price": 100}


async def test_plan_sparse_without_fuzzy_or_terms():
    conn = RecordingConn(row={"e0": False})
    plan = await sparse.plan_sparse(conn, "nupste", SearchFilters(), fuzzy=False)
    assert plan.missing == ["nupste"] and "fuzzy" not in render(conn.calls[0][0])
    empty = RecordingConn()
    assert (await sparse.plan_sparse(empty, "for the", SearchFilters())).terms == [] and empty.calls == []


async def test_search_sparse_adds_fuzzy_clauses():
    from app.schemas.query_models import SearchFilters

    conn = RecordingConn(rows=[{"product_id": 7, "score": 3.5}])
    hits = await sparse.search_sparse(conn, "kith hoddie lamp", SearchFilters(), 50,
                                      fuzzy_terms={"hoddie": 2, "lamp": 1})
    text = render(conn.calls[0][0])
    assert "search_text ||| %(q)s OR search_text ||| %(fz1)s::pdb.fuzzy(1) OR search_text ||| %(fz2)s::pdb.fuzzy(2)" in text
    assert conn.calls[0][1]["fz2"] == "hoddie" and conn.calls[0][1]["fz1"] == "lamp"
    assert hits == [Hit(7, 1, 3.5)]

    plain = RecordingConn(rows=[])
    await sparse.search_sparse(plain, "samba", SearchFilters(), 50)
    assert "fuzzy" not in render(plain.calls[0][0])


# ---------------------------------------------------------------------------
# hybrid_search decisions
# ---------------------------------------------------------------------------

def card(pid: int) -> ProductCard:
    return ProductCard(product_id=pid, title=f"P{pid}", price_min=Decimal(1), price_max=Decimal(1),
                       on_sale=False, in_stock=True, url="https://x")


class FakePool:
    @asynccontextmanager
    async def connection(self):
        yield object()


@pytest.fixture
def fake_search(monkeypatch):
    """Replace every I/O call in hybrid_search; tests set the hits each method returns."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    hybrid.get_settings.cache_clear()
    state = {"sparse": [], "dense": [], "plan": SparsePlan(terms=["q"], exact=["q"]), "calls": []}

    async def fake_plan(conn, text, filters, *, fuzzy=True):
        state["calls"].append(("plan", fuzzy))
        return state["plan"]

    async def fake_sparse(conn, text, filters, k, fuzzy_terms=None):
        state["calls"].append(("sparse", fuzzy_terms))
        return [Hit(p, i, 1.0) for i, p in enumerate(state["sparse"], start=1)]

    async def fake_dense(conn, vec, filters, k):
        state["calls"].append(("dense",))
        return [Hit(p, i, 0.1) for i, p in enumerate(state["dense"], start=1)]

    async def fake_embed(text):
        state["calls"].append(("embed",))
        state["embedded"] = text
        return [0.0]

    async def fake_cards(pool, ids):
        return {i: card(i) for i in ids}

    monkeypatch.setattr(hybrid, "plan_sparse", fake_plan)
    monkeypatch.setattr(hybrid, "search_sparse", fake_sparse)
    monkeypatch.setattr(hybrid, "search_dense", fake_dense)
    monkeypatch.setattr(hybrid.llm, "embed_query", fake_embed)
    monkeypatch.setattr(hybrid, "_fetch_cards", fake_cards)
    yield state
    hybrid.get_settings.cache_clear()


def ids(res):
    return [r.product.product_id for r in res.results]


async def test_code_query_uses_bm25_only_and_skips_embedding(fake_search):
    fake_search["sparse"], fake_search["dense"] = [1, 2], [9, 8]
    res = await hybrid.hybrid_search(FakePool(), SearchQuery(text="aaki4323", limit=10))
    assert ids(res) == [1, 2] and res.debug["route"] == "code:bm25"
    assert ("embed",) not in fake_search["calls"]
    assert not any(c[0] == "plan" for c in fake_search["calls"])      # no per-word checks for codes


async def test_unknown_code_falls_back_to_dense(fake_search):
    fake_search["sparse"], fake_search["dense"] = [], [9, 8]
    res = await hybrid.hybrid_search(FakePool(), SearchQuery(text="zz9999999", limit=10))
    assert ids(res) == [9, 8] and res.debug["route"] == "code:dense-fallback"


async def test_code_routing_can_be_switched_off(fake_search, monkeypatch):
    monkeypatch.setenv("ROUTE_CODES_TO_BM25", "false")
    hybrid.get_settings.cache_clear()
    fake_search["sparse"], fake_search["dense"] = [1], [9]
    res = await hybrid.hybrid_search(FakePool(), SearchQuery(text="aaki4323", limit=10))
    assert res.debug["route"] == "hybrid" and set(ids(res)) == {1, 9}


async def test_partial_coverage_lowers_bm25_weight(fake_search):
    # BM25 matched only "kith": its #1 (a pair of sunglasses) must not beat dense's #1.
    fake_search["plan"] = SparsePlan(terms=["kith", "hoddie"], exact=["kith"], missing=["hoddie"])
    fake_search["sparse"], fake_search["dense"] = [100, 101], [200, 201]
    res = await hybrid.hybrid_search(FakePool(), SearchQuery(text="kith hoddie", limit=4))
    assert res.debug["weights"] == {"bm25": 0.5, "dense": 1.0}
    assert ids(res)[:2] == [200, 201]


async def test_full_coverage_keeps_equal_weights(fake_search):
    fake_search["sparse"], fake_search["dense"] = [100], [200]
    res = await hybrid.hybrid_search(FakePool(), SearchQuery(text="samba", limit=4))
    assert res.debug["weights"] == {"bm25": 1.0, "dense": 1.0}
    assert ids(res) == [100, 200]           # tie broken by best rank, then id


async def test_fuzzy_terms_are_passed_to_bm25(fake_search):
    fake_search["plan"] = SparsePlan(terms=["nupste", "puffer"], exact=["puffer"], fuzzy={"nupste": 2})
    await hybrid.hybrid_search(FakePool(), SearchQuery(text="nupste puffer", limit=4))
    assert ("sparse", {"nupste": 2}) in fake_search["calls"]


async def test_coverage_weighting_can_be_switched_off(fake_search, monkeypatch):
    monkeypatch.setenv("BM25_COVERAGE_WEIGHTING", "false")
    monkeypatch.setenv("RRF_WEIGHT_DENSE", "1.5")
    hybrid.get_settings.cache_clear()
    fake_search["plan"] = SparsePlan(terms=["a1", "b1"], exact=["a1"], missing=["b1"])
    res = await hybrid.hybrid_search(FakePool(), SearchQuery(text="a1 b1", limit=4))
    assert res.debug["weights"] == {"bm25": 1.0, "dense": 1.5}


async def test_dense_text_is_embedded_instead_of_query_text(fake_search):
    await hybrid.hybrid_search(FakePool(), SearchQuery(text="black sneakers", limit=4), dense_text="I need black sneakers")
    assert fake_search["embedded"] == "I need black sneakers"
    await hybrid.hybrid_search(FakePool(), SearchQuery(text="black sneakers", limit=4))
    assert fake_search["embedded"] == "black sneakers"


async def test_single_modes_unchanged(fake_search):
    fake_search["sparse"], fake_search["dense"] = [1, 2], [9, 8]
    assert ids(await hybrid.hybrid_search(FakePool(), SearchQuery(text="samba", limit=10), "bm25")) == [1, 2]
    assert ids(await hybrid.hybrid_search(FakePool(), SearchQuery(text="samba", limit=10), "dense")) == [9, 8]


def test_pool_never_prepares_statements(monkeypatch):
    """ParadeDB's ||| fails under generic plans, which prepared statements can switch to."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from app.core.config import get_settings
    from app.core.db import create_pool
    get_settings.cache_clear()
    pool = create_pool()
    assert pool.kwargs["prepare_threshold"] is None and pool.kwargs["autocommit"] is True
    get_settings.cache_clear()
