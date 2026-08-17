"""
rate_limiter.py — respect free-tier rate limits.

Gemini free tier:  15 RPM, 1500 RPD
Groq free tier:    30 RPM, 14400 RPD
Claude:            pay-as-you-go (no hard limit, but we still track)

Tracks both per-minute (RPM) and per-day (RPD) windows. If the per-minute
budget is exhausted it blocks; if the *daily* budget is exhausted it raises
``DailyQuotaExceeded`` so the caller can fall back to another provider.
"""

import os
import time
from collections import defaultdict, deque
from threading import Lock

__all__ = ["wait_for_quota", "DailyQuotaExceeded", "snapshot",
           "register_provider", "known_providers", "LIMITS"]


class DailyQuotaExceeded(Exception):
    """Raised when a provider's requests-per-day budget is used up."""


_lock = Lock()

LIMITS = {
    "claude": {"rpm": 999, "rpd": 999_999},
    "gemini": {"rpm": 15, "rpd": 1500},
    "groq": {"rpm": 30, "rpd": 14_400},
    # OpenRouter free tier is conservative; exceeding it just falls through to
    # the next provider in the chain.
    "openrouter": {"rpm": 20, "rpd": 1000},
    # Google CSE free tier is 100 queries/day. Both bounds are configurable so
    # a paid tier can raise them without a code change.
    "google": {
        "rpm": int(os.getenv("GOOGLE_REQUESTS_PER_MINUTE", "10")),
        "rpd": int(os.getenv("GOOGLE_REQUESTS_PER_DAY", "90")),
    },
    # Mistral free ("La Plateforme" free tier) limits are account/model
    # specific and not publicly fixed, so these are deliberately CONSERVATIVE
    # placeholders, both overridable. They are not a claim about the real quota.
    "mistral": {
        "rpm": int(os.getenv("MISTRAL_REQUESTS_PER_MINUTE", "10")),
        "rpd": int(os.getenv("MISTRAL_REQUESTS_PER_DAY", "500")),
    },
}

# Quota state is DERIVED from LIMITS, never hand-maintained. Adding a provider
# to LIMITS is now sufficient — the previous KeyError('openrouter') /
# KeyError('google') bug came from parallel dicts that had to be kept in sync
# by hand. defaultdict also covers a provider registered at runtime.
_minute_history = defaultdict(deque, {name: deque() for name in LIMITS})
_day_history = defaultdict(deque, {name: deque() for name in LIMITS})


def register_provider(name: str, rpm: int, rpd: int) -> None:
    """Add/override a provider's limits at runtime; state is auto-created."""
    LIMITS[name] = {"rpm": int(rpm), "rpd": int(rpd)}
    _minute_history[name]      # touch so the deque exists immediately
    _day_history[name]


def known_providers() -> list:
    return sorted(LIMITS)


def wait_for_quota(model: str) -> None:
    """Block until calling ``model`` again is safe under its rate limits.

    Raises ``DailyQuotaExceeded`` if the daily request budget is gone.
    Sleeps OUTSIDE the lock so one throttled provider never stalls the others.
    """
    if model not in LIMITS:
        return
    while True:
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

            # Under the per-minute cap → register the call and go.
            if len(minute_hist) < limits["rpm"]:
                stamp = time.time()
                minute_hist.append(stamp)
                day_hist.append(stamp)
                return

            wait = 60 - (now - minute_hist[0]) + 0.1

        print(f"⏳ Rate limit reached for {model}, waiting {wait:.1f}s")
        time.sleep(max(wait, 0.1))


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
