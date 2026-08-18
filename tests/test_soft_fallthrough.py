#!/usr/bin/env python3
"""Rate-limit sleeps must fall through to an idle provider.

In the last production run Groq latched FAILED (404) and OpenRouter latched
RATE_LIMITED (429) correctly, but Mistral then carried all 72 calls and slept
45-48s at least six times while Gemini sat at 0 requests, status ENABLED. The
chain treated "rate limited, sleeping" as success instead of advancing.

A wait longer than MAX_PROVIDER_WAIT_SECONDS is now a SOFT failure: skip this
provider for THIS call and move down the chain. It must not latch — unlike the
404/429 hard latches, which stay exactly as they were.

Run:  python tests/test_soft_fallthrough.py
"""
import os
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_router as mr
import rate_limiter as rl

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  ✅ {msg}")


OK_RESPONSE = {"content": "x", "model_used": "m", "task_type": "t",
               "tokens_used": {"input": 1, "output": 1}, "cost_usd": 0.0,
               "fell_back": False}
ENV = {"GROQ_API_KEY": "g", "OPENROUTER_API_KEY": "o", "MISTRAL_API_KEY": "m",
       "GEMINI_API_KEY": "ge", "DAILY_CLAUDE_BUDGET_USD": "0",
       "ANTHROPIC_API_KEY": ""}
GROQ_404 = RuntimeError("404 The model `x` does not exist or you do not have "
                        "access to it.")
OPENROUTER_429 = RuntimeError("429 Too Many Requests")


class Chain:
    """Run call_model with every provider stubbed and the quota probe faked."""

    def __init__(self, waits=None, errors=None):
        self.calls = []
        self.slept = []
        # provider -> seconds it would sleep, or a callable returning that
        self.waits = waits or {}
        self.errors = errors or {}

    def _stub(self, name):
        def f(prompt, system, max_tokens, temperature, task_type, *a, **k):
            self.calls.append(name)
            err = self.errors.get(name)
            if err:
                raise err
            return dict(OK_RESPONSE)
        return f

    def __enter__(self):
        mr.reset_provider_stats()
        self._p = [
            patch.dict(os.environ, ENV),
            patch.object(mr, "_call_groq", self._stub("groq")),
            patch.object(mr, "_call_openrouter", self._stub("openrouter")),
            patch.object(mr, "_call_mistral", self._stub("mistral")),
            patch.object(mr, "_call_gemini", self._stub("gemini")),
            patch.object(mr, "wait_for_quota",
                         side_effect=lambda p: self.slept.append(p)),
            patch.object(mr, "seconds_until_available",
                         side_effect=self._wait_for),
            patch.object(mr, "get_daily_claude_spend", return_value=0.0),
            patch.object(mr, "get_monthly_claude_spend", return_value=0.0),
        ]
        for p in self._p:
            p.start()
        return self

    def _wait_for(self, provider):
        w = self.waits.get(provider, 0.0)
        return float(w() if callable(w) else w)

    def __exit__(self, *e):
        for p in reversed(self._p):
            p.stop()
        return False

    def run(self, task="first_pass_filter"):
        try:
            return mr.call_model(task, "prompt")
        except Exception as ex:
            return ex


# ══════════════════════════════════════════════════════════════════════════
print("\n1. Production scenario: groq 404, openrouter 429, mistral throttled")
# Mistral is free for its first call, then hits its per-minute cap — which is
# exactly what happened in production once it was carrying the whole scan.
_mistral_calls = {"n": 0}


def mistral_wait():
    _mistral_calls["n"] += 1
    return 0.0 if _mistral_calls["n"] == 1 else 47.0


with Chain(waits={"mistral": mistral_wait}, errors={"groq": GROQ_404,
                                                    "openrouter": OPENROUTER_429}) as c:
    c.run()                                    # candidate 1 sets both latches
    first = list(c.calls)
    for _ in range(11):                        # candidates 2-12
        c.run()

counts = Counter(c.calls)
assert first == ["groq", "openrouter", "mistral"], first
assert counts["groq"] == 1, counts           # hard latch, unchanged
assert counts["openrouter"] == 1, counts     # hard latch, unchanged
assert counts["mistral"] == 1, counts        # throttled after its first call
assert counts["gemini"] == 11, counts        # the idle provider does the work
ok(f"12 calls → groq 1 (latched), openrouter 1 (latched), "
   f"mistral 1 then throttle-skipped, gemini {counts['gemini']}")

# ══════════════════════════════════════════════════════════════════════════
print("\n2. A soft skip does NOT latch the provider")
with Chain(waits={"mistral": 47.0}) as c:
    c.waits["groq"] = 47.0
    c.run()
    assert c.calls == ["openrouter"], c.calls
    s = mr.provider_stats("groq")
    assert s["soft_skips"] == 1, s
    assert s["failed"] == 0 and s["429"] == 0 and s["404"] == 0, s
    assert s["config_error"] is False, s
    assert mr.provider_state("groq") == mr.ENABLED, mr.provider_state("groq")
    assert mr._provider_available("groq") is True, "soft skip disabled groq!"

    # ...and the very next call uses it again once the wait clears.
    c.waits["groq"] = 0.0
    c.run()
    assert c.calls == ["openrouter", "groq"], c.calls
ok("soft skip records no failure, keeps state ENABLED, recovers next call")

# ══════════════════════════════════════════════════════════════════════════
print("\n3. The 404 / 429 hard latches are unchanged")
with Chain(errors={"groq": GROQ_404, "openrouter": OPENROUTER_429}) as c:
    c.run()
    assert mr.provider_stats("groq")["config_error"] is True
    assert mr.provider_state("groq") == mr.FAILED
    assert mr.provider_state("openrouter") == mr.RATE_LIMITED
    assert mr.provider_stats("groq")["soft_skips"] == 0, "404 counted as soft"
    assert mr.provider_stats("openrouter")["soft_skips"] == 0
    before = len(c.calls)
    c.run()
    later = c.calls[before:]
    assert "groq" not in later and "openrouter" not in later, later
ok("hard latches still fire, still skip for the whole scan, still distinct")

# ══════════════════════════════════════════════════════════════════════════
print("\n4. A short wait is tolerated — we do not skip on every hiccup")
with Chain(waits={"groq": mr.MAX_PROVIDER_WAIT_SECONDS}) as c:
    c.run()
    assert c.calls == ["groq"], c.calls       # exactly at the limit: allowed
    assert mr.provider_stats("groq")["soft_skips"] == 0
with Chain(waits={"groq": mr.MAX_PROVIDER_WAIT_SECONDS + 0.1}) as c:
    c.run()
    assert c.calls == ["openrouter"], c.calls  # just over: skipped
ok(f"threshold is exclusive at {mr.MAX_PROVIDER_WAIT_SECONDS}s "
   f"(<= waits, > skips)")

# ══════════════════════════════════════════════════════════════════════════
print("\n5. All providers throttled → wait for the soonest, do not fail")
with Chain(waits={"groq": 55.0, "openrouter": 40.0, "mistral": 47.0,
                  "gemini": 30.0}) as c:
    res = c.run()
    assert not isinstance(res, Exception), res
    assert c.calls == ["gemini"], c.calls     # smallest wait wins
ok("every provider busy → takes the shortest wait (gemini 30s), no failure")

# A provider whose DAILY quota is gone (inf) is never chosen as "soonest".
with Chain(waits={"groq": float("inf"), "openrouter": float("inf"),
                  "mistral": float("inf"), "gemini": 25.0}) as c:
    c.run()
    assert c.calls == ["gemini"], c.calls
with Chain(waits={p: float("inf") for p in
                  ("groq", "openrouter", "mistral", "gemini")}) as c:
    res = c.run()
    assert isinstance(res, mr.ProvidersUnavailable), type(res)
    assert c.calls == [], c.calls
ok("daily-exhausted providers are never woken; all-exhausted preserves the "
   "candidate via ProvidersUnavailable")

# ══════════════════════════════════════════════════════════════════════════
print("\n6. seconds_until_available: read-only and matches wait_for_quota")
rl.register_provider("unittest_provider", rpm=2, rpd=100)
assert rl.seconds_until_available("unittest_provider") == 0.0
before = rl.snapshot()["unittest_provider"]["last_minute"]
rl.seconds_until_available("unittest_provider")
assert rl.snapshot()["unittest_provider"]["last_minute"] == before, \
    "the probe reserved a slot"
ok("probing reserves nothing")

rl.wait_for_quota("unittest_provider")
rl.wait_for_quota("unittest_provider")        # now at the 2/min cap
wait = rl.seconds_until_available("unittest_provider")
assert 55 < wait <= 60.2, wait
ok(f"at the per-minute cap the probe reports {wait:.1f}s without sleeping")

rl.register_provider("unittest_dry", rpm=5, rpd=1)
rl.wait_for_quota("unittest_dry")
assert rl.seconds_until_available("unittest_dry") == float("inf")
assert rl.seconds_until_available("not_a_provider") == 0.0
ok("daily-exhausted reports inf; an unknown provider reports 0.0")

# ══════════════════════════════════════════════════════════════════════════
print("\n7. Soft skips are visible in the health report")
with Chain(waits={"groq": 47.0}) as c:
    c.run()
    lines = "\n".join(mr.model_health_lines())
assert "throttle-skipped 1" in lines, lines
ok("MODEL HEALTH shows 'throttle-skipped N' for busy providers")

print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
