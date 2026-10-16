"""
Generation without OpenAI or a database: the LLM client and the search are faked.

    pytest tests/test_generation.py -q
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.chat import service
from app.generation import generate
from app.generation.context import ProductDetails, build_context, product_block
from app.generation.generate import has_price_text, validate
from app.generation.prompt import build_input
from app.generation.schemas import AnswerDraft, ComparisonDraft, ComparisonRowDraft, Pick, Turn
from app.schemas.product_models import ProductCard, RetrievedProduct, SearchResult
from app.schemas.query_models import SearchFilters


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def card(pid: int, **kw) -> ProductCard:
    base = dict(product_id=pid, title=f"Product {pid}", brand="adidas", price_min=Decimal("100"),
                price_max=Decimal("100"), on_sale=False, in_stock=True, sizes_available=["9", "10"],
                url=f"https://kith.com/p/{pid}", category="footwear", subcategory="low top sneakers",
                gender="mens")
    return ProductCard(**{**base, **kw})


CARDS = [card(1), card(2, title="Samba OG", collaborator="CLOT"), card(3, in_stock=False, sizes_available=[])]


def draft(**kw) -> AnswerDraft:
    base = dict(kind="recommend", message="These fit well.", picks=[Pick(product_id=1, reason="Classic.")],
                comparison=None, follow_up=None)
    return AnswerDraft(**{**base, **kw})


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

def test_product_block_has_catalog_facts_and_trims():
    c = card(7, price_min=Decimal("80"), price_max=Decimal("80"), on_sale=True, compare_at_price=Decimal("120"))
    d = ProductDetails(colorway="Core Black", materials=["suede"], bullets=["<b>Gum</b> sole"] + [f"b{i}" for i in range(10)])
    block = product_block(c, d)
    assert block.startswith('<product id="7">') and block.endswith("</product>")
    assert "Price: $80 (on sale, was $120) | In stock: yes | Sizes in stock: 9, 10" in block
    assert "Brand: adidas" in block and "Colorway: Core Black" in block and "Materials: suede" in block
    assert "(b)Gum(/b) sole" in block               # no tags can break out of the block
    assert block.count("\n- ") == 6                 # bullets capped


def test_build_context_handles_missing_details_and_ranges():
    ctx = build_context([card(1, price_min=Decimal("90"), price_max=Decimal("110.5"))], {})
    assert "Price: $90-$110.50" in ctx and "Details:" not in ctx


def test_build_input_layout():
    text = build_input("cheaper?", [("user", "black hoodie")], "<product id=\"1\">…</product>", {"max_price": 150})
    assert text.index("<conversation>") < text.index("filtered by: max_price: 150") < text.index("<products>") \
        < text.index("<message>\ncheaper?\n</message>")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_unknown_duplicate_and_extra_picks_are_dropped():
    d = draft(picks=[Pick(product_id=1, reason="a"), Pick(product_id=99, reason="b"),
                     Pick(product_id=1, reason="c"), Pick(product_id=2, reason="d"), Pick(product_id=3, reason="e")])
    a = validate(d, CARDS, max_picks=2)
    assert [p.product.product_id for p in a.products] == [1, 2]
    assert any("unknown product_id 99" in w for w in a.warnings)
    assert any("duplicate" in w for w in a.warnings) and any("beyond the first 2" in w for w in a.warnings)


def test_all_picks_invalid_falls_back_to_top_results():
    a = validate(draft(picks=[Pick(product_id=42, reason="x")]), CARDS, 5)
    assert a.source == "template" and [p.product.product_id for p in a.products] == [1, 2, 3]
    assert any("no valid picks" in w for w in a.warnings)


def test_compare_builds_table_with_catalog_columns():
    d = draft(kind="compare", picks=[Pick(product_id=1, reason="a"), Pick(product_id=3, reason="b")],
              comparison=ComparisonDraft(attributes=["Upper", "Sole"], rows=[
                  ComparisonRowDraft(product_id=1, values=["Leather", "Gum", "extra"]),
                  ComparisonRowDraft(product_id=3, values=["Suede"]),
                  ComparisonRowDraft(product_id=99, values=["?", "?"])]))
    a = validate(d, CARDS, 5)
    t = a.comparison
    assert t.columns == ["Upper", "Sole", "Price", "On sale", "Sizes in stock"]
    assert t.rows[0].values == ["Leather", "Gum", "$100.00", "No", "9, 10"]
    assert t.rows[1].values == ["Suede", "Not stated", "$100.00", "No", "Sold out"]
    assert len(t.rows) == 2


def test_compare_with_one_valid_product_becomes_recommend():
    d = draft(kind="compare", comparison=ComparisonDraft(attributes=["Upper"], rows=[
        ComparisonRowDraft(product_id=1, values=["Leather"])]))
    a = validate(d, CARDS, 5)
    assert a.kind == "recommend" and a.comparison is None and any("compare needs" in w for w in a.warnings)


def test_price_in_text_is_flagged():
    assert has_price_text("only $95 today") and has_price_text("about 120 dollars") and has_price_text("30% off")
    assert not has_price_text("the 990v6 in size 10") and not has_price_text("Samba OG")
    a = validate(draft(message="This one is $80 now."), CARDS, 5)
    assert any("price" in w for w in a.warnings)


def test_clarify_and_no_match():
    a = validate(draft(kind="clarify", message="For whom?", picks=[Pick(product_id=1, reason="x")]), CARDS, 5)
    assert a.kind == "clarify" and a.products == []
    nm = validate(draft(kind="no_match", message="We don't sell tuxedos.", picks=[]), CARDS, 5)
    assert nm.kind == "no_match" and nm.products == [] and nm.source == "llm"


def test_comparison_on_non_compare_is_ignored():
    d = draft(comparison=ComparisonDraft(attributes=["A"], rows=[]))
    a = validate(d, CARDS, 5)
    assert a.comparison is None and any("comparison ignored" in w for w in a.warnings)


# ---------------------------------------------------------------------------
# draft_answer: the LLM call
# ---------------------------------------------------------------------------

class FakeResponses:
    def __init__(self, result):
        self.result, self.calls = result, []

    async def parse(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(output_parsed=self.result, usage=SimpleNamespace(input_tokens=2000, output_tokens=200))


async def test_draft_answer_call(monkeypatch):
    fake = FakeResponses(draft())
    client = SimpleNamespace(responses=fake, with_options=lambda **kw: client)
    monkeypatch.setattr(generate, "get_openai_client", lambda: client)
    d = await generate.draft_answer("hi", [], "<product id=\"1\"></product>", {})
    assert d.answer.kind == "recommend" and d.input_tokens == 2000
    call = fake.calls[0]
    assert call["text_format"] is AnswerDraft and call["store"] is False
    assert call["reasoning"] == {"effort": "low"} and "<products>" in call["input"]


async def test_draft_answer_rejects_unparsed(monkeypatch):
    client = SimpleNamespace(responses=FakeResponses(None))
    client.with_options = lambda **kw: client
    monkeypatch.setattr(generate, "get_openai_client", lambda: client)
    with pytest.raises(ValueError):
        await generate.draft_answer("hi", [], "", {})


# ---------------------------------------------------------------------------
# chat service
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline(monkeypatch):
    state = {"qu": {"used": True, "inferred_filters": {"categories": ["footwear"]}}, "cards": CARDS,
             "draft": draft(), "exc": None, "search_args": None, "draft_args": None}

    async def fake_search(pool, message, filters, limit, history=None):
        state["search_args"] = dict(message=message, filters=filters, limit=limit, history=history)
        return SearchResult(mode="hybrid", results=[RetrievedProduct(product=c, score=1.0) for c in state["cards"]],
                            timings_ms={"total": 900.0}, debug={"qu": state["qu"]})

    async def fake_details(pool, ids):
        return {i: ProductDetails(bullets=["Gum sole"]) for i in ids}

    async def fake_draft(message, history, context, applied):
        state["draft_args"] = dict(message=message, history=history, context=context, applied=applied)
        if state["exc"]:
            raise state["exc"]
        return generate.Draft(state["draft"], 1500.0, 2000, 150)

    monkeypatch.setattr(service, "search", fake_search)
    monkeypatch.setattr(service, "load_details", fake_details)
    monkeypatch.setattr(service, "draft_answer", fake_draft)
    return state


async def test_chat_happy_path(pipeline):
    hist = [Turn(role="user", content="black hoodie"), Turn(role="assistant", content="Here are some.")]
    a = await service.chat(None, "something warmer", hist, SearchFilters(max_price=Decimal("200")))
    assert a.kind == "recommend" and a.products[0].product.product_id == 1 and a.source == "llm"
    assert pipeline["search_args"]["history"] == ["black hoodie"]           # QU sees user turns only
    assert pipeline["draft_args"]["history"] == [("user", "black hoodie"), ("assistant", "Here are some.")]
    assert pipeline["draft_args"]["applied"] == {"max_price": "200", "categories": ["footwear"]}
    assert "Gum sole" in pipeline["draft_args"]["context"]
    assert a.debug["retrieved"] == [1, 2, 3] and a.debug["tokens"] == {"input": 2000, "output": 150}
    assert {"search", "context", "generate", "total"} <= set(a.timings_ms)


async def test_chat_out_of_scope_skips_llm(pipeline):
    pipeline["qu"] = {"used": True, "out_of_scope": True}
    pipeline["exc"] = AssertionError("must not be called")
    a = await service.chat(None, "what's your return policy?")
    assert a.kind == "out_of_scope" and a.source == "template" and a.products == []


async def test_chat_no_results(pipeline):
    pipeline["cards"] = []
    a = await service.chat(None, "unicorn boots")
    assert a.kind == "no_match" and a.source == "template" and "footwear" in a.message


async def test_chat_generation_failure_shows_results(pipeline):
    pipeline["exc"] = TimeoutError("slow")
    a = await service.chat(None, "sneakers")
    assert a.source == "template" and len(a.products) == 3 and "generation failed" in a.warnings[0]


async def test_history_is_trimmed(pipeline, monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_TURNS", "2")
    from app.core.config import get_settings
    get_settings.cache_clear()
    hist = [Turn(role="user", content=f"m{i}") for i in range(5)]
    await service.chat(None, "next", hist)
    assert pipeline["search_args"]["history"] == ["m3", "m4"]
