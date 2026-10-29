"""
Application settings, read from environment variables (and .env in development).

Every model name lives here, one setting per task, so switching a model is a
configuration change. Any field can be overridden by an environment variable
of the same name in upper case, e.g. GENERATION_MODEL=gpt-6.1-sol.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

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
    # The judge runs offline through the Batch API, so latency doesn't matter: give it room
    # to check each claim against the product blocks. Empty value to omit it.
    judge_reasoning_effort: Optional[str] = "medium"

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

    # --- Query understanding (app/query) ----------------------------------------
    # One LLM call turns the shopper's message into a QueryPlan: a cleaner
    # search text, inferred filters and an intent. If it is off, slow or fails,
    # search runs on the raw text with the rules above.
    query_understanding: bool = True
    query_timeout_seconds: float = 8.0     # past this, fall back to the raw query
    query_max_retries: int = 1             # a shopper is waiting: retry once, not five times
    # Dense weight in the fusion by intent (BM25 stays at rrf_weight_bm25).
    # "specific": a named product, where exact words matter most.
    # "explore": a need or a vibe, where meaning matters most.
    qu_dense_weight_specific: float = 0.75
    qu_dense_weight_explore: float = 1.5
    # If inferred filters leave fewer results than this, search again without them.
    qu_relax_min_results: int = 5
    # Subcategory is filled for only ~2/3 of products, and the parser's names are
    # uneven ("sneakers" vs "low top sneakers"), so an inferred subcategory filter
    # hides good results. Off: the plan still records it, but search ignores it.
    qu_subcategory_filters: bool = False
    # Passed as reasoning={"effort": ...} to trade depth for latency. "low" cut the
    # LLM step from 2.5 s to 1.7 s (p50) with no measurable loss in retrieval quality.
    # Set to an empty value (QUERY_REASONING_EFFORT=) for models that don't support it.
    query_reasoning_effort: Optional[str] = "low"
    # What vector search embeds when query understanding ran: "original" (the
    # shopper's message; for a follow-up, the plan's self-contained standalone_query)
    # or "rewrite" (the plan's keyword search_text). BM25
    # always gets the rewrite: fixed spelling and expansion help exact matching,
    # while a keyword list loses the sentence meaning embeddings are good at.
    qu_dense_text: Literal["original", "rewrite"] = "original"

    # --- Generation (app/generation, app/chat) -----------------------------------
    generation_candidates: int = 8         # products retrieved and shown to the model
    generation_max_picks: int = 5          # products the answer may recommend
    generation_timeout_seconds: float = 25.0
    generation_max_retries: int = 1
    generation_reasoning_effort: Optional[str] = "low"   # empty value to omit for models without it
    chat_history_turns: int = 6            # earlier turns passed to the model

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
