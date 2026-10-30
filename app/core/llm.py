"""
The one place the project talks to OpenAI.

Both the offline embedder and the live API embed through these functions, so
product vectors and query vectors always come from the same model with the
same number of dimensions. Switching provider means changing this file only.

Retries: every call goes through app.core.retry (exponential backoff with full
jitter, Retry-After respected) with the SDK's own retries turned off, so the
two never stack. Live calls also take a concurrency slot and a time budget
(call_llm): a shopper is waiting, so a retry that can't finish in time isn't
started and the caller falls back instead.
"""

from __future__ import annotations

import asyncio
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Awaitable, Callable, Optional, TypeVar

from openai import AsyncOpenAI

from app.core import tracing
from app.core.config import get_settings
from app.core.retry import with_retries

T = TypeVar("T")

# OpenAI accepts at most 8,192 tokens per input. Product texts are far shorter,
# but a runaway description shouldn't fail a whole batch. ~4 characters per
# token for English text, so this cap stays safely under the limit.
MAX_INPUT_CHARS = 24_000


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    """One shared client per process. It retries rate limits and server errors itself."""
    s = get_settings()
    return AsyncOpenAI(
        api_key=s.openai_api_key.get_secret_value(),
        timeout=s.openai_timeout_seconds,
        max_retries=s.openai_max_retries,
    )


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    tokens: int
    model: str


# ---------------------------------------------------------------------------
# Live calls: concurrency slot + retries + time budget
# ---------------------------------------------------------------------------

# One semaphore per event loop (tests and CLIs each run their own loop).
_slots: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def _llm_slot() -> asyncio.Semaphore:
    """Caps model calls in flight from this process, so a traffic spike queues here
    instead of turning into a burst of 429s from OpenAI."""
    loop = asyncio.get_running_loop()
    sem = _slots.get(loop)
    if sem is None:
        sem = _slots[loop] = asyncio.Semaphore(get_settings().llm_max_concurrency)
    return sem


async def call_llm(make_call: Callable[[], Awaitable[T]], *, budget_seconds: float, attempts: int) -> T:
    """
    A model call a shopper is waiting for. Waiting for a slot, every attempt and
    every backoff all count against budget_seconds; past it, asyncio.TimeoutError
    (or the last API error) is raised and the caller falls back.
    `make_call` must use a client with max_retries=0.
    """
    s = get_settings()
    deadline = time.monotonic() + budget_seconds

    async def run() -> T:
        async with _llm_slot():
            return await with_retries(make_call, attempts=attempts, base=s.llm_retry_base_seconds,
                                      cap=s.llm_retry_cap_seconds, deadline=deadline)

    return await asyncio.wait_for(run(), timeout=budget_seconds + 1)   # backstop if a timeout misfires


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

async def embed_texts(texts: list[str], *, attempts: Optional[int] = None,
                      deadline: Optional[float] = None) -> EmbeddingResult:
    """
    Embed a batch of texts, returned in the same order as given.
    Retries with full jitter: `attempts` defaults to 1 + openai_max_retries (bulk
    embedding); `deadline` (a time.monotonic() value) bounds a live call.
    Raises openai.BadRequestError if any input is rejected, and ValueError if
    the vectors don't have the configured number of dimensions.
    """
    s = get_settings()
    if not texts:
        return EmbeddingResult(vectors=[], tokens=0, model=s.embedding_model)

    client = get_openai_client().with_options(max_retries=0)
    response = await with_retries(
        lambda: client.embeddings.create(
            model=s.embedding_model,
            input=[t[:MAX_INPUT_CHARS] for t in texts],
            dimensions=s.embedding_dimensions,
            encoding_format="float",
        ),
        attempts=attempts or 1 + s.openai_max_retries,
        base=s.llm_retry_base_seconds,
        cap=s.llm_retry_cap_seconds,
        deadline=deadline,
    )

    vectors = [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
    if len(vectors) != len(texts):
        raise ValueError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
    bad = {len(v) for v in vectors} - {s.embedding_dimensions}
    if bad:
        raise ValueError(
            f"expected {s.embedding_dimensions}-dim vectors from {s.embedding_model}, got {sorted(bad)}"
        )

    tokens = response.usage.total_tokens if response.usage else 0
    return EmbeddingResult(vectors=vectors, tokens=tokens, model=s.embedding_model)


# Shoppers (and eval runs) repeat queries; an identical query always gives the
# same vector, so recent ones are kept in memory. Keyed by model and dimensions
# too, so a config change never returns a stale vector.
_QUERY_CACHE_SIZE = 1024
_query_cache: OrderedDict[tuple[str, int, str], list[float]] = OrderedDict()


def clear_query_cache() -> None:
    """Forget cached query vectors (evals use this so every mode pays for its own embedding)."""
    _query_cache.clear()


async def embed_query(text: str) -> list[float]:
    """Embed one search query, with the same model and dimensions as the products."""
    s = get_settings()
    key = (s.embedding_model, s.embedding_dimensions, text)
    with tracing.observe("embed-query", as_type="embedding", model=s.embedding_model,
                         input=tracing.mask_pii(text)) as obs:
        if key in _query_cache:
            _query_cache.move_to_end(key)
            obs.update(metadata={"cached": True})
            return _query_cache[key]
        # A shopper is waiting: few attempts, bounded by the query budget.
        result = await embed_texts([text], attempts=1 + s.query_max_retries,
                                   deadline=time.monotonic() + s.query_timeout_seconds)
        obs.update(metadata={"cached": False, "dimensions": s.embedding_dimensions},
                   usage_details={"input": result.tokens})
    vec = result.vectors[0]
    _query_cache[key] = vec
    if len(_query_cache) > _QUERY_CACHE_SIZE:
        _query_cache.popitem(last=False)
    return vec
