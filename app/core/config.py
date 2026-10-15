"""
Application settings, read from environment variables (and .env in development).

Every model name lives here, one setting per task, so switching a model is a
configuration change. Any field can be overridden by an environment variable
of the same name in upper case, e.g. GENERATION_MODEL=gpt-6.1-sol.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # .env also holds POSTGRES_* for Docker; ignore those here
    )

    # --- Database -----------------------------------------------------------
    database_url: str = Field(description="postgresql://user:password@host:port/db")

    # --- OpenAI -------------------------------------------------------------
    openai_api_key: SecretStr
    openai_timeout_seconds: float = 30.0
    openai_max_retries: int = 5            # SDK retries 429 / 5xx / connection errors with backoff

    # Embeddings. The dimension count must match the vector(N) column in
    # db/schema.sql; the embedder checks this before doing any work.
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 768

    # Chat models, one per task (see the model-selection notes in the README).
    query_model: str = "gpt-6-luna"          # message -> structured search query
    generation_model: str = "gpt-6-luna"     # answers and comparisons
    judge_model: str = "gpt-6.1-sol"         # offline evals only

    # --- Retrieval ----------------------------------------------------------
    retrieval_candidates: int = 50     # results taken from EACH method before fusion
    rrf_k: int = 60                    # RRF constant; 60 is the value from the original paper
    rrf_weight_bm25: float = 1.0       # how much each method's list counts in the fusion
    rrf_weight_dense: float = 1.0

    # Rule-based query handling (see app/retrieval/query_analysis.py). Each
    # can be switched off with an environment variable, so the eval can
    # measure what each one is worth, e.g. ROUTE_CODES_TO_BM25=false.
    route_codes_to_bm25: bool = True       # style codes / SKUs: BM25 answers alone
    bm25_fuzzy_fallback: bool = True       # words with no exact match are retried with typos allowed
    bm25_coverage_weighting: bool = True   # BM25 counts less when it matches only part of the query

    # HNSW scans only `ef_search` candidates and filters AFTER the scan, so a
    # query can never return more than ef_search rows (pgvector's default is 40).
    # Keep this above retrieval_candidates.
    hnsw_ef_search: int = 100
    # pgvector 0.8+ can keep scanning until enough rows pass the filters.
    # Ignored automatically on older pgvector versions.
    hnsw_iterative_scan: Literal["off", "relaxed_order", "strict_order"] = "relaxed_order"


@lru_cache
def get_settings() -> Settings:
    """Settings are read once and cached for the life of the process."""
    return Settings()