"""
Latency features: the query embedding runs alongside query understanding, in-flight
requests are shared, and the answer streams. No database, no OpenAI.

    pytest tests/test_latency.py -q
"""

from __future__ import annotations

import asyncio

import pytest

from app.core import llm


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()
    llm.clear_query_cache()


class FakeEmbeddings:
    """embed_texts stand-in: counts calls, can be slow, can fail first."""

    def __init__(self, delay: float = 0.0, fail_first: bool = False):
        self.calls, self.delay, self.fail_first = [], delay, fail_first

    async def __call__(self, texts, **kw):
        self.calls.append(list(texts))
        await asyncio.sleep(self.delay)
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("embedding service down")
        return llm.EmbeddingResult(vectors=[[float(len(texts[0]))] * 4], tokens=3, model="m")


# --- embedding: prefetch and in-flight sharing ------------------------------------

async def test_a_prefetched_embedding_is_used_not_repeated(env):
    fake = FakeEmbeddings(delay=0.05)
    env.setattr(llm, "embed_texts", fake)
    llm.start_embed_query("black hoodie")
    vec = await llm.embed_query("black hoodie")              # still in flight: waits for it
    assert vec == [12.0] * 4 and len(fake.calls) == 1
    assert await llm.embed_query("black hoodie") == vec and len(fake.calls) == 1   # now cached


async def test_concurrent_requests_for_one_text_share_a_call(env):
    fake = FakeEmbeddings(delay=0.05)
    env.setattr(llm, "embed_texts", fake)
    a, b, c = await asyncio.gather(*(llm.embed_query("samba") for _ in range(3)))
    assert a == b == c and len(fake.calls) == 1


async def test_a_failed_prefetch_is_retried_by_the_real_lookup(env):
    fake = FakeEmbeddings(fail_first=True)
    env.setattr(llm, "embed_texts", fake)
    llm.start_embed_query("nuptse")
    await asyncio.sleep(0.01)                                  # the prefetch fails on its own
    assert await llm.embed_query("nuptse") == [6.0] * 4 and len(fake.calls) == 2


async def test_one_caller_giving_up_does_not_cancel_the_shared_call(env):
    fake = FakeEmbeddings(delay=0.05)
    env.setattr(llm, "embed_texts", fake)
    first = asyncio.ensure_future(llm.embed_query("gazelle"))
    await asyncio.sleep(0.01)
    first.cancel()
    assert await llm.embed_query("gazelle") == [7.0] * 4 and len(fake.calls) == 1


async def test_warm_up_never_raises(env):
    async def down(texts, **kw):
        raise RuntimeError("no network")
    env.setattr(llm, "embed_texts", down)
    await llm.warm_up()


# --- search: when the embedding starts early ------------------------------------------

@pytest.fixture
def search_pipeline(env):
    """Faked understanding and retrieval that record whether the embedding had started."""
    from app.query import search as search_mod
    from app.query import understanding
    from tests.test_query_understanding import VOCAB, plan, result_with

    env.setenv("PREFETCH_QUERY_EMBEDDING", "true")
    from app.core.config import get_settings
    get_settings.cache_clear()
    fake = FakeEmbeddings(delay=0.02)
    env.setattr(llm, "embed_texts", fake)
    state = {"embed_started_before_plan": None, "fake": fake}

    async def fake_vocab(pool):
        return VOCAB

    async def fake_understand(message, vocab, history=None):
        await asyncio.sleep(0)                                 # let a started prefetch begin
        state["embed_started_before_plan"] = bool(fake.calls)
        return understanding.Understanding(plan=plan(), latency_ms=1.0)

    async def fake_hybrid(pool, query, mode="hybrid", weights=None, dense_text=None):
        await llm.embed_query(dense_text or query.text)
        return result_with(10)

    env.setattr(search_mod, "load_vocabulary", fake_vocab)
    env.setattr(search_mod, "understand", fake_understand)
    env.setattr(search_mod, "hybrid_search", fake_hybrid)
    return state


async def test_search_embeds_the_message_while_the_llm_reads_it(search_pipeline):
    from app.query import search as search_mod
    await search_mod.search(None, "cozy hoodie")
    assert search_pipeline["embed_started_before_plan"] is True
    assert search_pipeline["fake"].calls == [["cozy hoodie"]]          # one call, reused by retrieval


async def test_follow_ups_wait_for_the_plan(search_pipeline):
    from app.query import search as search_mod
    await search_mod.search(None, "something cheaper", history=["black hoodie"])
    assert search_pipeline["embed_started_before_plan"] is False


async def test_prefetch_can_be_switched_off(search_pipeline, env):
    from app.query import search as search_mod
    from app.core.config import get_settings
    env.setenv("PREFETCH_QUERY_EMBEDDING", "false")
    get_settings.cache_clear()
    await search_mod.search(None, "cozy hoodie")
    assert search_pipeline["embed_started_before_plan"] is False


# --- prompt caching ---------------------------------------------------------------------

def test_cached_prompt_tokens_are_counted_apart():
    from types import SimpleNamespace
    usage = SimpleNamespace(input_tokens=1700, output_tokens=90,
                            input_tokens_details=SimpleNamespace(cached_tokens=1536))
    assert llm.usage_counts(usage) == (1700, 1536, 90)
    assert llm.trace_usage(1700, 1536, 90) == {"input": 164, "input_cached_tokens": 1536, "output": 90}
    assert llm.usage_counts(SimpleNamespace(input_tokens=5, output_tokens=1)) == (5, 0, 1)   # no details
    assert llm.usage_counts(None) == (0, 0, 0)


async def test_query_understanding_sends_a_stable_prompt_cache_key(env):
    from types import SimpleNamespace
    from app.query import understanding
    from tests.test_query_understanding import VOCAB, plan
    seen = {}

    async def parse(**kw):
        seen.update(kw)
        return SimpleNamespace(output_parsed=plan(), usage=SimpleNamespace(
            input_tokens=1700, output_tokens=90, input_tokens_details=SimpleNamespace(cached_tokens=1536)))
    client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    client.with_options = lambda **kw: client
    env.setattr(understanding, "get_openai_client", lambda: client)
    understanding.clear_plan_cache()
    u = await understanding.understand("black hoodie", VOCAB)
    assert seen["prompt_cache_key"] == f"kith-qu-{understanding.PROMPT_VERSION}"
    assert (u.input_tokens, u.cached_tokens, u.output_tokens) == (1700, 1536, 90)
    understanding.clear_plan_cache()


# --- streaming: reading the message out of partial JSON ------------------------------------

@pytest.mark.parametrize("buffer, expected", [
    ('', None),
    ('{"kind":"recommend"', None),
    ('{"kind":"recommend","message":"', ''),
    ('{"kind":"recommend","message":"Here are tw', 'Here are tw'),
    ('{"kind":"recommend","message":"Caf\\u00', 'Caf'),                      # half an escape: wait
    ('{"kind":"recommend","message":"Café.","picks":[{"product_id":1', 'Café.'),
    ('not json at all', None),
])
def test_partial_message(buffer, expected):
    from app.generation.generate import partial_message
    assert partial_message(buffer) == expected


# --- streaming: generation -----------------------------------------------------------------

def _draft_json(message: str = "Two good hoodies for you.") -> str:
    import json
    return json.dumps({"kind": "recommend", "message": message, "picks": [{"product_id": 2, "reason": "Fits."}],
                       "comparison": None, "follow_up": None})


class FakeStream:
    """client.responses.stream(...) stand-in: emits the JSON in chunks, can fail."""

    def __init__(self, text: str, chunk: int = 7, fail_after: int | None = None, exc: Exception | None = None):
        self.text, self.chunk, self.fail_after, self.exc = text, chunk, fail_after, exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        from types import SimpleNamespace
        pieces = [self.text[i:i + self.chunk] for i in range(0, len(self.text), self.chunk)]
        yield SimpleNamespace(type="response.created")
        for n, piece in enumerate(pieces):
            if self.fail_after is not None and n == self.fail_after:
                raise self.exc
            await asyncio.sleep(0)
            yield SimpleNamespace(type="response.output_text.delta", delta=piece)

    async def get_final_response(self):
        from types import SimpleNamespace
        from app.generation.schemas import AnswerDraft
        return SimpleNamespace(output_parsed=AnswerDraft.model_validate_json(self.text),
                               usage=SimpleNamespace(input_tokens=1900, output_tokens=200,
                                                     input_tokens_details=SimpleNamespace(cached_tokens=0)))


def stream_client(env, *streams):
    """A fake OpenAI client whose responses.stream returns the given streams in turn."""
    from types import SimpleNamespace
    from app.generation import generate
    calls = []

    def stream(**kw):
        calls.append(kw)
        return streams[len(calls) - 1]
    client = SimpleNamespace(responses=SimpleNamespace(stream=stream))
    client.with_options = lambda **kw: client
    env.setattr(generate, "get_openai_client", lambda: client)
    env.setenv("LLM_RETRY_BASE_SECONDS", "0.001")
    from app.core.config import get_settings
    get_settings.cache_clear()
    return calls


async def test_stream_answer_sends_the_message_in_pieces(env):
    from app.generation.generate import stream_answer
    calls = stream_client(env, FakeStream(_draft_json()))
    pieces: list[str] = []

    async def on_delta(t):
        pieces.append(t)
    draft = await stream_answer("hoodie", [], '<product id="2">\nTitle: Hoodie\n</product>', {}, on_delta)
    assert "".join(pieces) == "Two good hoodies for you." and len(pieces) > 1
    assert draft.answer.picks[0].product_id == 2 and draft.first_token_ms is not None
    assert calls[0]["prompt_cache_key"].startswith("kith-gen-") and calls[0]["text_format"].__name__ == "AnswerDraft"


async def test_stream_retries_only_before_the_first_word(env):
    from app.generation.generate import stream_answer
    from tests.test_retry import status_error
    stream_client(env, FakeStream(_draft_json(), fail_after=0, exc=status_error(429)), FakeStream(_draft_json()))
    pieces: list[str] = []

    async def on_delta(t):
        pieces.append(t)
    draft = await stream_answer("hoodie", [], "", {}, on_delta)              # failed before any text: retried
    assert "".join(pieces) == "Two good hoodies for you." and draft.answer.kind == "recommend"


async def test_stream_failing_after_text_is_raised_not_retried(env):
    from app.generation.generate import stream_answer
    from tests.test_retry import status_error
    calls = stream_client(env, FakeStream(_draft_json(), fail_after=6, exc=status_error(503)), FakeStream(_draft_json()))

    async def on_delta(t):
        pass
    with pytest.raises(Exception):
        await stream_answer("hoodie", [], "", {}, on_delta)
    assert len(calls) == 1


# --- streaming: the chat turn ----------------------------------------------------------------

@pytest.fixture
def streaming_turn(env):
    """chat_stream with a faked search and a fake streaming model."""
    from app.chat import service
    from tests.test_generation import CARDS

    async def fake_search(pool, message, filters, limit, history=None):
        from app.schemas.product_models import RetrievedProduct, SearchResult
        return SearchResult(mode="hybrid", results=[RetrievedProduct(product=c, score=1.0) for c in CARDS],
                            timings_ms={"total": 5.0}, debug={"qu": {"used": True}})

    async def fake_details(pool, ids):
        return {}
    env.setattr(service, "search", fake_search)
    env.setattr(service, "load_details", fake_details)
    return env


async def test_chat_stream_reports_progress_then_returns_the_validated_answer(streaming_turn):
    from app.chat.service import chat_stream
    stream_client(streaming_turn, FakeStream(_draft_json()))
    events: list[tuple] = []

    async def emit(e, d):
        events.append((e, d))
    answer = await chat_stream(None, "hoodie", emit=emit)
    assert events[:2] == [("status", "searching"), ("status", "writing")]
    assert "".join(d for e, d in events if e == "delta") == answer.message == "Two good hoodies for you."
    assert [p.product.product_id for p in answer.products] == [2]
    assert answer.timings_ms["first_token"] <= answer.timings_ms["total"]


async def test_chat_stream_falls_back_when_the_model_fails_mid_answer(streaming_turn):
    from app.chat.service import chat_stream
    from tests.test_retry import status_error
    stream_client(streaming_turn, FakeStream(_draft_json(), fail_after=5, exc=status_error(503)))
    events: list[tuple] = []

    async def emit(e, d):
        events.append((e, d))
    answer = await chat_stream(None, "hoodie", emit=emit)
    assert any(e == "delta" for e, _ in events)                       # some text had been shown
    assert answer.source == "template" and answer.warnings[0].startswith("generation failed")


# --- streaming: SSE ---------------------------------------------------------------------------

def parse_sse(text: str) -> list[tuple[str, dict]]:
    import json
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


async def test_sse_stream_event_order_and_errors():
    from app.api.sse import sse_stream

    async def ok(emit):
        await emit("status", "searching")
        await emit("delta", "Hi")
        return {"kind": "recommend"}
    events = parse_sse("".join([e async for e in sse_stream(ok, "rid1")]))
    assert events == [("meta", {"request_id": "rid1"}), ("status", {"stage": "searching"}),
                      ("delta", {"text": "Hi"}), ("answer", {"kind": "recommend"})]

    async def boom(emit):
        raise RuntimeError("secret detail")
    events = parse_sse("".join([e async for e in sse_stream(boom, "rid2")]))
    assert events[-1] == ("error", {"error": {"code": "internal_error", "message": "Something went wrong. Please try again.",
                                              "request_id": "rid2"}})


async def test_closing_the_stream_cancels_the_work():
    from app.api.sse import sse_stream
    state = {"cancelled": False}

    async def slow(emit):
        await emit("status", "writing")
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
    gen = sse_stream(slow, "rid3")
    assert (await gen.__anext__()).startswith("event: meta")
    assert (await gen.__anext__()).startswith("event: status")
    await gen.aclose()                                             # the client disconnected
    await asyncio.sleep(0.01)
    assert state["cancelled"] is True


async def test_stream_endpoint(env):
    import httpx
    from app.api.main import create_app
    from app.api.routes import chat as chat_route
    from app.generation.schemas import ChatAnswer

    async def fake_chat_stream(pool, message, history=None, filters=None, *, emit):
        await emit("status", "searching")
        await emit("status", "writing")
        for piece in ("Two good ", "hoodies."):
            await emit("delta", piece)
        return ChatAnswer(kind="recommend", message="Two good hoodies.", timings_ms={"first_token": 900.0})
    env.setattr(chat_route, "chat_stream", fake_chat_stream)
    app = create_app(use_lifespan=False)
    app.state.pool = object()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.9", 1))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/v1/chat/stream", json={"message": "hoodie"}, headers={"X-Request-ID": "abc-1"})
        bad = await c.post("/v1/chat/stream", json={"message": ""})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [e for e, _ in events] == ["meta", "status", "status", "delta", "delta", "answer"]
    answer = events[-1][1]
    assert answer["request_id"] == "abc-1" and answer["message"] == "Two good hoodies." and answer["debug"] is None
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_request"   # before streaming: JSON


# --- the benchmark report ---------------------------------------------------------------------

def test_latency_report(env):
    from app.core.config import get_settings
    from evals.run_latency import report
    env.setenv("PREFETCH_QUERY_EMBEDDING", "true")
    get_settings.cache_clear()
    rows = [
        {"case": "a", "kind": "recommend", "wall_ms": 4000,
         "timings": {"search": 2600, "context": 5, "generate": 1400, "first_token": 3100, "total": 4000},
         "search_timings": {"understand": 2400, "embed_query": 2, "bm25_search": 40, "dense_search": 30}},
        {"case": "b", "kind": "out_of_scope", "wall_ms": 2500, "timings": {"search": 2500, "total": 2500},
         "search_timings": {"understand": 2300}},
        {"case": "c", "error": "TimeoutError: slow"},
    ]
    text = report(rows, "baseline")
    assert "2 turns (1 failed)" in text and "embedding prefetch on" in text
    assert "| first words | 1 | 3,100 | 3,100 | 3,100 |" in text
    assert "| understand | 2 | 2,300 | 2,400 | 2,400 |" in text
    assert "`c`: TimeoutError: slow" in text
