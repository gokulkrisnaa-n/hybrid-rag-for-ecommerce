"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_real_tracing(monkeypatch):
    """Settings also read .env: with Langfuse keys there, tests would send fake traces to the
    real project. Tracing is off in every test unless a test turns it on with its own client.

    The search pipeline starts embedding the message in the background; with the pipeline
    faked, that would be a real call to OpenAI. Off too, unless a test turns it on."""
    from app.core import llm, tracing
    from app.core.config import get_settings
    monkeypatch.setenv("TRACING_ENABLED", "false")
    monkeypatch.setenv("PREFETCH_QUERY_EMBEDDING", "false")
    llm.clear_query_cache()
    get_settings.cache_clear()
    tracing.reset_client()
    yield
    tracing.reset_client()
    get_settings.cache_clear()
    