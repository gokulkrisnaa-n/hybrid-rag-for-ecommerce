"""
Application settings, read from environment variables (and .env in development).

Every model name lives here, one setting per task, so switching a model is a
configuration change. Any field can be overridden by an environment variable
of the same name in upper case, e.g. GENERATION_MODEL=gpt-6.1-sol.
"""

from __future__ import annotations

from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    """Settings are read once and cached for the life of the process."""
    return Settings()