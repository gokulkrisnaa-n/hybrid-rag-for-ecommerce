"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_real_tracing(monkeypatch):
    """Settings also read .env: with Langfuse keys there, tests would send fake traces to the
    real project. Tracing is off in every test unless a test turns it on with its own client."""
    from app.core import tracing
    from app.core.config import get_settings
    monkeypatch.setenv("TRACING_ENABLED", "false")
    get_settings.cache_clear()
    tracing.reset_client()
    yield
    tracing.reset_client()
    get_settings.cache_clear()
