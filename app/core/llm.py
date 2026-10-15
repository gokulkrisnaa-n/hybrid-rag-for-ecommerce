"""
The one place the project talks to OpenAI.

Both the offline embedder and the live API embed through these functions, so
product vectors and query vectors always come from the same model with the
same number of dimensions. Switching provider means changing this file only.

Chat helpers (query understanding, generation) get added here when those
steps are built.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

from openai import AsyncOpenAI

from app.core.config import get_settings

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


async def embed_texts(texts: list[str]) -> EmbeddingResult:
    """
    Embed a batch of texts, returned in the same order as given.
    Raises openai.BadRequestError if any input is rejected, and ValueError if
    the vectors don't have the configured number of dimensions.
    """
    s = get_settings()
    if not texts:
        return EmbeddingResult(vectors=[], tokens=0, model=s.embedding_model)

    response = await get_openai_client().embeddings.create(
        model=s.embedding_model,
        input=[t[:MAX_INPUT_CHARS] for t in texts],
        dimensions=s.embedding_dimensions,
        encoding_format="float",
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


async def embed_query(text: str) -> list[float]:
    """Embed one search query, with the same model and dimensions as the products."""
    s = get_settings()
    key = (s.embedding_model, s.embedding_dimensions, text)
    if key in _query_cache:
        _query_cache.move_to_end(key)
        return _query_cache[key]
    vec = (await embed_texts([text])).vectors[0]
    _query_cache[key] = vec
    if len(_query_cache) > _QUERY_CACHE_SIZE:
        _query_cache.popitem(last=False)
    return vec