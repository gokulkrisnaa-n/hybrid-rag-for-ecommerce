"""
Retries (full jitter, Retry-After, deadlines), live-call budgets and the concurrency cap. No network.

    pytest tests/test_retry.py -q
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import httpx
import openai
import pytest

from app.core import llm
from app.core.retry import backoff_delay, is_retryable, retry_after_seconds, with_retries

REQ = httpx.Request("POST", "https://api.openai.com/v1/responses")


def status_error(status: int, headers: dict | None = None) -> openai.APIStatusError:
    resp = httpx.Response(status, headers=headers or {}, request=REQ)
    cls = {429: openai.RateLimitError, 400: openai.BadRequestError, 401: openai.AuthenticationError,
           404: openai.NotFoundError}.get(status, openai.InternalServerError if status >= 500 else openai.APIStatusError)
    return cls("error", response=resp, body=None)


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class Calls:
    """A call that fails with the given errors first, then returns 'ok'."""

    def __init__(self, *errors):
        self.errors, self.n = list(errors), 0

    async def __call__(self):
        self.n += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


class Sleeps:
    def __init__(self):
        self.waits = []

    async def __call__(self, s):
        self.waits.append(s)


# --- backoff ------------------------------------------------------------------

@pytest.mark.parametrize("attempt, r, expected", [
    (0, 1.0, 0.5), (1, 1.0, 1.0), (3, 1.0, 4.0), (10, 1.0, 8.0),   # ceiling doubles, then caps
    (2, 0.0, 0.0), (2, 0.5, 1.0),                                   # full jitter: anywhere from 0
])
def test_backoff_delay_is_full_jitter_under_a_capped_ceiling(attempt, r, expected):
    assert backoff_delay(attempt, 0.5, 8.0, rand=lambda: r) == pytest.approx(expected)


@pytest.mark.parametrize("exc, expected", [
    (status_error(429), True), (status_error(500), True), (status_error(503), True),
    (status_error(408), True), (status_error(409), True),
    (status_error(400), False), (status_error(401), False), (status_error(404), False),
    (openai.APIConnectionError(request=REQ), True), (openai.APITimeoutError(request=REQ), True),
    (ValueError("bad output"), False),
])
def test_is_retryable(exc, expected):
    assert is_retryable(exc) is expected


def test_retry_after_headers():
    assert retry_after_seconds(status_error(429, {"retry-after-ms": "1500"})) == 1.5
    assert retry_after_seconds(status_error(429, {"retry-after": "3"})) == 3.0
    now = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
    date = format_datetime(now + timedelta(seconds=7), usegmt=True)
    assert retry_after_seconds(status_error(429, {"retry-after": date}), now=now) == pytest.approx(7)
    assert retry_after_seconds(status_error(429)) is None
    assert retry_after_seconds(ValueError("x")) is None


# --- with_retries -------------------------------------------------------------

async def test_retries_then_succeeds_with_jittered_waits():
    call, sleeps = Calls(status_error(429), status_error(503)), Sleeps()
    assert await with_retries(call, attempts=3, base=0.5, cap=8, sleep=sleeps, rand=lambda: 0.5) == "ok"
    assert call.n == 3 and sleeps.waits == [0.25, 0.5]


async def test_non_retryable_error_is_raised_at_once():
    call, sleeps = Calls(status_error(400)), Sleeps()
    with pytest.raises(openai.BadRequestError):
        await with_retries(call, attempts=5, sleep=sleeps)
    assert call.n == 1 and sleeps.waits == []


async def test_last_error_is_raised_when_attempts_run_out():
    call = Calls(status_error(429), status_error(429), status_error(429))
    with pytest.raises(openai.RateLimitError):
        await with_retries(call, attempts=2, sleep=Sleeps(), rand=lambda: 0.0)
    assert call.n == 2


async def test_server_retry_after_wins_over_backoff():
    call, sleeps = Calls(status_error(429, {"retry-after-ms": "1200"})), Sleeps()
    await with_retries(call, attempts=2, sleep=sleeps, rand=lambda: 1.0)
    assert sleeps.waits == [1.2]


async def test_very_long_retry_after_is_not_waited_for():
    call = Calls(status_error(429, {"retry-after": "120"}))
    with pytest.raises(openai.RateLimitError):
        await with_retries(call, attempts=3, sleep=Sleeps())
    assert call.n == 1


async def test_no_retry_that_would_pass_the_deadline():
    now = [100.0]
    call, sleeps = Calls(status_error(429, {"retry-after": "2"})), Sleeps()
    with pytest.raises(openai.RateLimitError):
        await with_retries(call, attempts=3, deadline=101.0, sleep=sleeps, clock=lambda: now[0])
    assert call.n == 1 and sleeps.waits == []


async def test_attempts_must_be_positive():
    with pytest.raises(ValueError):
        await with_retries(Calls(), attempts=0)


# --- live calls: budget and concurrency ------------------------------------------

async def test_call_llm_gives_up_at_the_budget():
    async def slow():
        await asyncio.sleep(5)
    with pytest.raises(asyncio.TimeoutError):
        await llm.call_llm(slow, budget_seconds=0.05, attempts=1)


async def test_call_llm_caps_calls_in_flight(monkeypatch):
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "2")
    llm._slots.clear()
    in_flight, peak = 0, 0

    async def call():
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return "ok"

    results = await asyncio.gather(*(llm.call_llm(call, budget_seconds=5, attempts=1) for _ in range(6)))
    assert results == ["ok"] * 6 and peak == 2
    llm._slots.clear()


async def test_embeddings_retry_with_sdk_retries_off(monkeypatch):
    monkeypatch.setenv("LLM_RETRY_BASE_SECONDS", "0.001")
    seen = {}

    class FakeEmbeddings:
        def __init__(self):
            self.calls = 0

        async def create(self, **kw):
            self.calls += 1
            if self.calls == 1:
                raise status_error(429)
            data = [type("D", (), {"embedding": [0.0] * 768, "index": i})() for i in range(len(kw["input"]))]
            return type("R", (), {"data": data, "usage": None})()

    fake = FakeEmbeddings()

    class FakeClient:
        embeddings = fake

        def with_options(self, **kw):
            seen.update(kw)
            return self

    monkeypatch.setattr(llm, "get_openai_client", lambda: FakeClient())
    result = await llm.embed_texts(["a", "b"])
    assert len(result.vectors) == 2 and fake.calls == 2 and seen == {"max_retries": 0}
