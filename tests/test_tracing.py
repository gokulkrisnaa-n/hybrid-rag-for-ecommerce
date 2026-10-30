"""
Tracing with the real Langfuse SDK, exporting spans to memory instead of the network.

    pytest tests/test_tracing.py -q
"""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.core import tracing
from app.query import understanding
from app.query.plan import QueryPlan
from app.query.vocabulary import Vocabulary

RID = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


@pytest.fixture
def spans(env):
    """Tracing on, with a real Langfuse client that exports to memory. Returns a function
    that flushes and gives {name: [span, ...]}."""
    from langfuse import Langfuse
    exporter = InMemorySpanExporter()
    client = Langfuse(public_key=f"pk-lf-test-{uuid.uuid4().hex}", secret_key="sk-lf-test",
                      base_url="http://127.0.0.1:9", span_exporter=exporter)
    scores: list[dict] = []
    client.create_score = lambda **kw: scores.append(kw)          # scores go over HTTP, not OTel
    env.setattr(tracing, "_client", client)

    def collect() -> dict[str, list]:
        client.flush()
        out: dict[str, list] = {}
        for s in exporter.get_finished_spans():
            out.setdefault(s.name, []).append(s)
        return out
    collect.scores = scores
    collect.client = client
    return collect


def attrs(span) -> dict:
    return dict(span.attributes)


def trace_hex(span) -> str:
    return format(span.context.trace_id, "032x")


def parent_of(span, by_name: dict[str, list]):
    pid = span.parent.span_id if span.parent else None
    for name, items in by_name.items():
        for s in items:
            if s.context.span_id == pid:
                return name
    return None


# --- masking ---------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("mail me at jane.doe@example.com", "mail me at [email]"),
    ("call +1 (555) 123-4567 or 555.123.4567", "call [phone] or [phone]"),
    ("card 4111 1111 1111 1111 please", "card [card] please"),
    ("card 4111111111111112", "card 4111111111111112"),               # fails Luhn: not a card
    ("order #10482, size 10.5, under $150", "order #10482, size 10.5, under $150"),
    ("product 8286760829056", "product 8286760829056"),
    ("style AI1203A844-100", "style AI1203A844-100"),
    (None, None), ("", ""),
])
def test_mask_pii(text, expected):
    assert tracing.mask_pii(text) == expected


# --- tracing off -------------------------------------------------------------------

async def test_everything_is_a_noop_without_keys(env):
    assert not tracing.tracing_active()
    with tracing.request_trace("chat", request_id=RID, input={"m": "x"}) as root:
        with tracing.observe("step", as_type="generation", model="m") as step:
            step.update(output="ignored")
        root.score("answer-fallback", 0, data_type="BOOLEAN")
    assert root is tracing.NOOP and not root.active
    assert tracing.record_feedback(RID, thumbs_up=True) is False


def test_trace_ids_follow_request_ids(env):
    assert tracing.trace_id_for(RID) == RID
    derived = tracing.trace_id_for("client-id-42")
    assert len(derived) == 32 and derived == tracing.trace_id_for("client-id-42") and derived != RID


# --- tracing on --------------------------------------------------------------------

async def test_request_trace_nests_steps_under_the_request_id(spans):
    async def retrieve(n):
        with tracing.observe(f"retriever-{n}", as_type="retriever") as r:
            await asyncio.sleep(0.005)
            r.update(output={"hits": n})

    with tracing.request_trace("chat", request_id=RID, input={"message": "hi"}, session_id="sess-1",
                               tags=["chat", "qu:qu-v6"], metadata={"route": "chat"}) as root:
        with tracing.observe("answer", as_type="generation", model="gpt-6-luna") as gen:
            gen.update(output={"kind": "recommend"}, usage_details={"input": 1900, "output": 200})
        await asyncio.gather(retrieve(1), retrieve(2))         # parallel steps keep the parent
        root.update(output={"kind": "recommend"})

    got = spans()
    assert {trace_hex(s) for items in got.values() for s in items} == {RID}
    for name in ("answer", "retriever-1", "retriever-2"):
        assert parent_of(got[name][0], got) == "chat"
    a = attrs(got["answer"][0])
    assert a["langfuse.observation.type"] == "generation"
    assert a["langfuse.observation.model.name"] == "gpt-6-luna"
    assert json.loads(a["langfuse.observation.usage_details"]) == {"input": 1900, "output": 200}
    root_attrs = attrs(got["chat"][0])
    assert root_attrs["session.id"] == "sess-1" and root_attrs["langfuse.trace.name"] == "chat"
    assert tuple(root_attrs["langfuse.trace.tags"]) == ("chat", "qu:qu-v6")
    assert json.loads(root_attrs["langfuse.observation.output"]) == {"kind": "recommend"}


async def test_an_error_marks_the_step_and_still_raises(spans):
    with pytest.raises(TimeoutError):
        with tracing.request_trace("chat", request_id=RID):
            with tracing.observe("understand-query", as_type="generation"):
                raise TimeoutError("model too slow")
    a = attrs(spans()["understand-query"][0])
    assert a["langfuse.observation.level"] == "ERROR"
    assert a["langfuse.observation.status_message"] == "TimeoutError: model too slow"


async def test_a_broken_tracer_never_breaks_the_request(env):
    class Broken:
        def start_as_current_observation(self, **kw):
            raise RuntimeError("tracing backend down")
    env.setattr(tracing, "_client", Broken())
    ran = False
    with tracing.observe("step") as obs:
        ran = True
        obs.update(output="x")
    assert ran and obs is tracing.NOOP


def test_feedback_is_one_replaceable_boolean_score(spans):
    assert tracing.record_feedback(RID, thumbs_up=False, comment="wrong size, call me at 555-123-4567")
    assert tracing.record_feedback(RID, thumbs_up=True)
    first, second = spans.scores
    assert first == {"name": "user-thumbs", "value": 0, "data_type": "BOOLEAN", "trace_id": RID,
                     "score_id": f"{RID}-user-thumbs", "comment": "wrong size, call me at [phone]"}
    assert second["value"] == 1 and second["score_id"] == first["score_id"]       # replaces, not adds


# --- a real step: query understanding ------------------------------------------------

VOCAB = Vocabulary(categories={"apparel": ["hoodies"]}, brands={"kith"})


def a_plan() -> QueryPlan:
    return QueryPlan(search_text="hoodie", standalone_query="a hoodie", intent="explore", audience="adult",
                     categories=["apparel"], subcategories=[], genders=[], brands=[], exclude_brands=[],
                     min_price=None, max_price=None, on_sale_only=False, compare_items=[], out_of_scope=False)


async def test_query_understanding_is_a_generation_then_a_cached_span(spans, env):
    async def parse(**kw):
        return SimpleNamespace(output_parsed=a_plan(), usage=SimpleNamespace(input_tokens=900, output_tokens=60))
    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    client.with_options = lambda **kw: client
    env.setattr(understanding, "get_openai_client", lambda: client)
    understanding.clear_plan_cache()

    await understanding.understand("hoodie for jane@example.com", VOCAB)
    await understanding.understand("hoodie for jane@example.com", VOCAB)       # cache hit
    first, second = spans()["understand-query"]
    a, b = attrs(first), attrs(second)
    assert a["langfuse.observation.type"] == "generation" and a["langfuse.observation.model.name"] == "gpt-6-luna"
    assert json.loads(a["langfuse.observation.usage_details"]) == {"input": 900, "output": 60}
    assert "[email]" in a["langfuse.observation.input"] and "jane@" not in a["langfuse.observation.input"]
    assert b["langfuse.observation.type"] == "span" and b["langfuse.observation.metadata.cached"] == "true"
    understanding.clear_plan_cache()


# --- API: chat trace, automatic scores, feedback endpoint ----------------------------------

async def call(app, method, path, **kw):
    transport = httpx.ASGITransport(app=app, client=("203.0.113.7", 1234))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        return await c.request(method, path, **kw)


def make_app():
    from app.api.main import create_app
    app = create_app(use_lifespan=False)
    app.state.pool = object()
    return app


async def test_chat_request_is_one_trace_with_automatic_scores(spans, env):
    from app.api.routes import chat as chat_route
    from app.generation.schemas import ChatAnswer
    scored: list[tuple] = []
    env.setattr(tracing.Observation, "score",
                lambda self, name, value, *, data_type, comment=None: scored.append((name, value, data_type)))

    async def fake_chat(pool, message, history=None, filters=None, include_context=False):
        with tracing.observe("answer", as_type="generation", model="gpt-6-luna"):
            pass
        return ChatAnswer(kind="recommend", message="Here you go.", warnings=["dropped unknown product_id 9"])

    env.setattr(chat_route, "chat", fake_chat)
    r = await call(make_app(), "POST", "/v1/chat", headers={"X-Request-ID": RID},
                   json={"message": "email me at a@b.co", "session_id": "conv-7"})
    assert r.status_code == 200 and r.json()["request_id"] == RID
    got = spans()
    root = attrs(got["chat"][0])
    assert trace_hex(got["chat"][0]) == RID and root["session.id"] == "conv-7"
    assert "[email]" in root["langfuse.observation.input"] and "a@b.co" not in root["langfuse.observation.input"]
    assert parent_of(got["answer"][0], got) == "chat"
    assert scored == [("answer-fallback", 0, "BOOLEAN"), ("answer-warnings", 1, "NUMERIC"),
                      ("answer-kind", "recommend", "CATEGORICAL")]


@pytest.mark.parametrize("answer_kwargs, fallback", [
    ({"kind": "recommend", "message": "x", "warnings": ["generation failed: TimeoutError"], "source": "template"}, True),
    ({"kind": "error", "message": "x"}, True),
    ({"kind": "out_of_scope", "message": "x", "source": "template"}, False),
    ({"kind": "recommend", "message": "x", "warnings": ["price or discount written in the text"]}, False),
])
def test_is_fallback(answer_kwargs, fallback):
    from app.api.routes.chat import is_fallback
    from app.generation.schemas import ChatAnswer
    assert is_fallback(ChatAnswer(**answer_kwargs)) is fallback


async def test_feedback_endpoint(spans):
    r = await call(make_app(), "POST", "/v1/feedback", json={"request_id": RID, "rating": -1, "comment": "not boots"})
    assert r.status_code == 202 and r.json() == {"request_id": RID, "status": "recorded"}
    assert spans.scores[0]["value"] == 0 and spans.scores[0]["comment"] == "not boots"


async def test_feedback_without_tracing_is_503(env):
    r = await call(make_app(), "POST", "/v1/feedback", json={"request_id": RID, "rating": 1})
    assert r.status_code == 503 and r.json()["error"]["code"] == "tracing_disabled"


@pytest.mark.parametrize("body", [
    {"request_id": RID, "rating": 0},
    {"request_id": "bad id\n", "rating": 1},
    {"request_id": RID, "rating": 1, "comment": "x" * 501},
])
async def test_feedback_validation(env, body):
    r = await call(make_app(), "POST", "/v1/feedback", json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


# --- the whole chat pipeline: the trace tree a real request produces ------------------------

async def test_full_chat_pipeline_trace_tree(spans, env):
    """Only the database and the models are faked; every traced step runs for real."""
    from contextlib import asynccontextmanager
    from decimal import Decimal

    from app.chat import service
    from app.core import llm
    from app.generation import generate
    from app.generation.schemas import AnswerDraft, Pick
    from app.query import search as search_mod
    from app.retrieval import hybrid
    from app.retrieval.sparse import Hit, SparsePlan
    from app.schemas.product_models import ProductCard

    class Pool:
        @asynccontextmanager
        async def connection(self):
            yield object()

    def card(pid):
        return ProductCard(product_id=pid, title=f"Hoodie {pid}", brand="Kith", price_min=Decimal("100"),
                           price_max=Decimal("100"), on_sale=False, in_stock=True, url=f"https://kith.com/{pid}")

    async def plan_sparse(conn, text, filters, fuzzy=True):
        return SparsePlan(terms=["hoodie"], exact=["hoodie"])

    async def search_sparse(conn, text, filters, k, fuzzy_terms=None):
        return [Hit(i, i, 10.0 - i) for i in range(1, 7)]        # 6 hits: no relaxation

    async def search_dense(conn, vec, filters, k):
        return [Hit(i, i - 1, 0.1 * i) for i in range(2, 8)]

    async def fetch_cards(pool, ids):
        return {i: card(i) for i in ids}

    async def embed_texts(texts, **kw):
        return llm.EmbeddingResult(vectors=[[0.0] * 768], tokens=7, model="text-embedding-3-small")

    async def load_vocabulary(pool):
        return VOCAB

    async def load_details(pool, ids):
        return {}

    qu_client = SimpleNamespace(responses=SimpleNamespace(parse=None))
    qu_client.with_options = lambda **kw: qu_client

    async def qu_parse(**kw):
        return SimpleNamespace(output_parsed=a_plan(), usage=SimpleNamespace(input_tokens=900, output_tokens=60))
    qu_client.responses.parse = qu_parse

    gen_client = SimpleNamespace(responses=SimpleNamespace(parse=None))
    gen_client.with_options = lambda **kw: gen_client

    async def gen_parse(**kw):
        draft = AnswerDraft(kind="recommend", message="Two hoodies.", picks=[Pick(product_id=2, reason="Fits.")],
                            comparison=None, follow_up=None)
        return SimpleNamespace(output_parsed=draft, usage=SimpleNamespace(input_tokens=1900, output_tokens=200))
    gen_client.responses.parse = gen_parse

    for mod, name, fake in [(hybrid, "plan_sparse", plan_sparse), (hybrid, "search_sparse", search_sparse),
                            (hybrid, "search_dense", search_dense), (hybrid, "_fetch_cards", fetch_cards),
                            (llm, "embed_texts", embed_texts), (search_mod, "load_vocabulary", load_vocabulary),
                            (service, "load_details", load_details),
                            (understanding, "get_openai_client", lambda: qu_client),
                            (generate, "get_openai_client", lambda: gen_client)]:
        env.setattr(mod, name, fake)
    understanding.clear_plan_cache()
    llm.clear_query_cache()

    with tracing.request_trace("chat", request_id=RID, input={"message": "hoodie"}):
        answer = await service.chat(Pool(), "a hoodie please")
    assert answer.kind == "recommend" and [p.product.product_id for p in answer.products] == [2]

    got = spans()
    tree = {name: parent_of(items[0], got) for name, items in got.items()}
    assert tree == {
        "chat": None,
        "search-products": "chat",
        "understand-query": "search-products",
        "resolve-filters": "search-products",
        "retrieve-hybrid": "search-products",
        "retrieve-bm25": "retrieve-hybrid",
        "retrieve-dense": "retrieve-hybrid",
        "embed-query": "retrieve-dense",
        "load-details": "chat",
        "generate-answer": "chat",
        "validate-answer": "chat",
    }
    types = {name: attrs(items[0])["langfuse.observation.type"] for name, items in got.items()}
    assert types["understand-query"] == "generation" and types["generate-answer"] == "generation"
    assert types["embed-query"] == "embedding" and types["retrieve-bm25"] == types["retrieve-dense"] == "retriever"
    assert types["validate-answer"] == "guardrail" and types["search-products"] == "chain"
    assert json.loads(attrs(got["embed-query"][0])["langfuse.observation.usage_details"]) == {"input": 7}
    answer = attrs(got["generate-answer"][0])
    assert json.loads(answer["langfuse.observation.input"]) == [{"role": "user", "content": "a hoodie please"}]
    assert "Hoodie 2" in answer["langfuse.observation.metadata.products"]
    understanding.clear_plan_cache()
    llm.clear_query_cache()
