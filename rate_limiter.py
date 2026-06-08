"""
rate_limiter.py — respect free-tier rate limits.

Gemini free tier:  15 RPM, 1500 RPD
Groq free tier:    30 RPM, 14400 RPD
Claude:            pay-as-you-go (no hard limit, but we still track)

Tracks both per-minute (RPM) and per-day (RPD) windows. If the per-minute
budget is exhausted it blocks; if the *daily* budget is exhausted it raises
``DailyQuotaExceeded`` so the caller can fall back to another provider.
"""

import time
from collections import deque
from threading import Lock

__all__ = ["wait_for_quota", "DailyQuotaExceeded", "snapshot"]


class DailyQuotaExceeded(Exception):
    """Raised when a provider's requests-per-day budget is used up."""


_minute_history = {"claude": deque(), "gemini": deque(), "groq": deque()}
_day_history = {"claude": deque(), "gemini": deque(), "groq": deque()}
_lock = Lock()

LIMITS = {
    "claude": {"rpm": 999, "rpd": 999_999},
    "gemini": {"rpm": 15, "rpd": 1500},
    "groq": {"rpm": 30, "rpd": 14_400},
}


def wait_for_quota(model: str) -> None:
    """Block until calling ``model`` again is safe under its rate limits.

    Raises ``DailyQuotaExceeded`` if the daily request budget is gone.
    """
    if model not in LIMITS:
        return
    with _lock:
        now = time.time()
        limits = LIMITS[model]
        minute_hist = _minute_history[model]
        day_hist = _day_history[model]

        # Expire entries outside their windows.
        while minute_hist and minute_hist[0] < now - 60:
            minute_hist.popleft()
        while day_hist and day_hist[0] < now - 86_400:
            day_hist.popleft()

        # Daily cap — fail loudly so caller can fall back.
        if len(day_hist) >= limits["rpd"]:
            raise DailyQuotaExceeded(
                f"{model} daily request budget ({limits['rpd']}) exhausted"
            )

        # Per-minute cap — block until the oldest call ages out.
        if len(minute_hist) >= limits["rpm"]:
            wait = 60 - (now - minute_hist[0]) + 0.1
            if wait > 0:
                print(f"⏳ Rate limit reached for {model}, waiting {wait:.1f}s")
                time.sleep(wait)
                now = time.time()
                while minute_hist and minute_hist[0] < now - 60:
                    minute_hist.popleft()

        stamp = time.time()
        minute_hist.append(stamp)
        day_hist.append(stamp)


def snapshot() -> dict:
    """Return current usage counts per provider (for diagnostics)."""
    now = time.time()
    out = {}
    with _lock:
        for model in LIMITS:
            minute = sum(1 for t in _minute_history[model] if t > now - 60)
            day = sum(1 for t in _day_history[model] if t > now - 86_400)
            out[model] = {
                "last_minute": minute,
                "rpm_limit": LIMITS[model]["rpm"],
                "today": day,
                "rpd_limit": LIMITS[model]["rpd"],
            }
    return out
