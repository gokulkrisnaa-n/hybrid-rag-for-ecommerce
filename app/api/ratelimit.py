"""
A sliding-window rate limit, in memory, per client key (IP address).

/v1/chat calls a paid model, so one visitor shouldn't be able to run up the
bill. This lives in the process: with several API instances each keeps its
own count, which is fine for one container. A shared limit would need Redis
or the gateway in front (e.g. Azure API Management).

The 429 carries Retry-After; clients should wait at least that long, and
back off with full jitter if they keep getting 429s.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Callable


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic,
                 max_keys: int = 10_000):
        self.limit, self.window, self.clock, self.max_keys = limit, window_seconds, clock, max_keys
        self._hits: dict[str, deque[float]] = {}

    def check(self, key: str) -> tuple[bool, int]:
        """(allowed, retry_after_seconds). Records the hit when allowed. limit <= 0 allows everything."""
        if self.limit <= 0:
            return True, 0
        now = self.clock()
        hits = self._hits.get(key)
        if hits is None:
            if len(self._hits) >= self.max_keys:
                self._prune(now)
            hits = self._hits[key] = deque()
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return False, max(1, math.ceil(self.window - (now - hits[0])))
        hits.append(now)
        return True, 0

    def _prune(self, now: float) -> None:
        """Forget clients whose hits have all expired, so memory stays bounded."""
        for key in [k for k, h in self._hits.items() if not h or now - h[-1] >= self.window]:
            del self._hits[key]
