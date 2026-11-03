"""Per-feature rate limits and daily caps (PRD_BACKEND §4.12, §6.5).

In-process and per worker, which matches the single-worker Render service. Per-IP limit
first (sliding window), then a global cap per UTC day.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime, timezone

from fastapi import Request

from app.errors import APIError

_PERIODS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def parse_rate(rate: str) -> tuple[int, int]:
    """'10/minute' -> (10, 60)."""
    count, _, period = rate.partition("/")
    return int(count), _PERIODS[period.strip().rstrip("s")]


def client_ip(request: Request) -> str:
    """The address the nearest proxy saw (rightmost X-Forwarded-For entry), else the peer."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


class FeatureLimiter:
    def __init__(
        self,
        rate: str,
        daily_cap: int,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], str] = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    ) -> None:
        self.limit, self.window = parse_rate(rate)
        self.daily_cap = daily_cap
        self._clock = clock
        self._today = today
        self._hits: dict[str, deque[float]] = {}
        self._day = ""
        self._day_count = 0
        self._lock = threading.Lock()

    def check(self, ip: str) -> None:
        """Count one request or raise 429 `rate_limited` / `daily_cap_reached`."""
        with self._lock:
            now = self._clock()
            hits = self._hits.setdefault(ip, deque())
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                raise APIError(429, "rate_limited", "Too many requests; try again shortly")
            day = self._today()
            if day != self._day:
                self._day, self._day_count = day, 0
                # Drop idle IPs once a day so the table does not grow without bound.
                self._hits = {k: v for k, v in self._hits.items() if v}
                hits = self._hits.setdefault(ip, deque())
            if self._day_count >= self.daily_cap:
                raise APIError(429, "daily_cap_reached", "Daily limit reached; try again tomorrow")
            self._day_count += 1
            hits.append(now)
