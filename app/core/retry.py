"""
Retries with exponential backoff and full jitter, for calls to OpenAI.

The OpenAI SDK retries on its own, but it waits min(0.5 s * 2^n, 8 s) minus
only 0-25% jitter, so many requests that hit a rate limit together also retry
together and hit it again. Full jitter waits a random time between 0 and the
exponential ceiling, which spreads the retries out:

    delay = random(0, min(cap, base * 2^attempt))

If the server says how long to wait (Retry-After / retry-after-ms), that wins.
A deadline bounds the whole thing: a shopper is waiting, so a retry that
can't finish in time isn't attempted and the caller falls back instead.

Calls wrapped here must have the SDK's own retries off (max_retries=0), or the
two would stack.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Awaitable, Callable, Optional, TypeVar

import openai

log = logging.getLogger("retry")

T = TypeVar("T")

RETRY_STATUSES = {408, 409, 429}
MAX_RETRY_AFTER_SECONDS = 60.0      # a server asking for longer is effectively "not now"


def backoff_delay(attempt: int, base: float, cap: float, rand: Callable[[], float] = random.random) -> float:
    """Full jitter: uniform in [0, min(cap, base * 2^attempt)]. attempt is 0 for the first retry."""
    return rand() * min(cap, base * (2 ** min(attempt, 30)))


def is_retryable(exc: BaseException) -> bool:
    """Rate limits, server errors, timeouts and dropped connections. Not 400/401/403/404/422."""
    if isinstance(exc, openai.APIConnectionError):          # includes APITimeoutError
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in RETRY_STATUSES or exc.status_code >= 500
    return False


def retry_after_seconds(exc: BaseException, now: Optional[datetime] = None) -> Optional[float]:
    """The server's requested wait, from retry-after-ms or Retry-After (seconds or an HTTP date)."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            return max(float(ms) / 1000, 0.0)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max((when - (now or datetime.now(timezone.utc))).total_seconds(), 0.0)


async def with_retries(
    call: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base: float = 0.5,
    cap: float = 8.0,
    deadline: Optional[float] = None,
    retryable: Callable[[BaseException], bool] = is_retryable,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """
    Run `call` up to `attempts` times. Retries only errors `retryable` accepts.
    `deadline` is a clock() value: no wait that would end after it is started;
    the last error is raised instead. Raises the last error when attempts run out.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001 - re-raised below unless retried
            if attempt == attempts - 1 or not retryable(exc):
                raise
            wait = retry_after_seconds(exc)
            if wait is None:
                wait = backoff_delay(attempt, base, cap, rand)
            elif wait > MAX_RETRY_AFTER_SECONDS:
                raise
            if deadline is not None and clock() + wait >= deadline:
                log.info("not retrying %s: %.2f s wait would pass the deadline", type(exc).__name__, wait)
                raise
            log.info("retry %d/%d after %s in %.2f s", attempt + 1, attempts - 1, type(exc).__name__, wait)
            await sleep(wait)
    raise AssertionError("unreachable")     # pragma: no cover
