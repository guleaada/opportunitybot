#!/usr/bin/env python3
"""The scan-level latch must hold when operations run concurrently.

The scan loop is sequential today, but the latch is shared mutable module
state and the project already guards its other shared state with a Lock
(rate_limiter, cost_tracker, database). These tests pin the behaviour so a
future concurrent scan cannot silently reintroduce the 68-wasted-calls bug.

Run:  python tests/test_latch_concurrency.py
"""
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_router as mr

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  ✅ {msg}")


GROQ_404 = "Error code: 404 - {'error': {'message': 'The model `x` does not exist"
ENV = {"GROQ_API_KEY": "g", "OPENROUTER_API_KEY": "o", "MISTRAL_API_KEY": "m",
       "GEMINI_API_KEY": "ge", "DAILY_CLAUDE_BUDGET_USD": "0",
       "ANTHROPIC_API_KEY": ""}
OK_RESPONSE = {"content": "x", "model_used": "m", "task_type": "t",
               "tokens_used": {"input": 1, "output": 1}, "cost_usd": 0.0,
               "fell_back": False}
OPERATIONS = ["clean_html", "first_pass_filter", "classify_opportunity",
              "scam_detection", "deep_eligibility", "final_scoring"]

calls = []
calls_lock = threading.Lock()


def stub(name, error=None, delay=False):
    def f(prompt, system, max_tokens, temperature, task_type, *a, **k):
        with calls_lock:
            calls.append(name)
        if delay:
            # Widen the window between "chain built" and "dispatch returns",
            # which is exactly where a race would slip through.
            threading.Event().wait(0.01)
        if error:
            raise RuntimeError(error)
        return dict(OK_RESPONSE)
    return f


# ══════════════════════════════════════════════════════════════════════════
print("\n1. 12 threads x 6 operations, groq 404 on first contact")
calls.clear()
mr.reset_provider_stats()
with patch.dict(os.environ, ENV), \
     patch.object(mr, "_call_groq", stub("groq", GROQ_404, delay=True)), \
     patch.object(mr, "_call_openrouter", stub("openrouter", "429 Too Many Requests")), \
     patch.object(mr, "_call_mistral", stub("mistral")), \
     patch.object(mr, "_call_gemini", stub("gemini")), \
     patch.object(mr, "wait_for_quota"), \
     patch.object(mr, "get_daily_claude_spend", return_value=0.0), \
     patch.object(mr, "get_monthly_claude_spend", return_value=0.0):

    def candidate(_):
        for task in OPERATIONS:
            try:
                mr.call_model(task, "prompt")
            except mr.ProvidersUnavailable:
                pass

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(candidate, range(12)))

    counts = Counter(calls)
    groq_stats = mr.provider_stats("groq")
    state = mr.provider_state("groq")

total_slots = 12 * len(OPERATIONS)
# Threads already in flight when the first 404 lands may also reach groq.
# What must NOT happen is groq being retried for the whole scan (72 slots).
assert counts["groq"] <= 12, f"groq called {counts['groq']}x of {total_slots}"
assert groq_stats["config_error"] is True
assert state == mr.FAILED
assert counts["mistral"] >= total_slots - counts["groq"], counts
ok(f"{total_slots} operation slots → groq attempted {counts['groq']}x "
   f"(never re-armed), latched FAILED, mistral served {counts['mistral']}")

# Counters must be internally consistent — no lost updates under the lock.
assert groq_stats["requests"] == counts["groq"], (groq_stats, counts)
assert groq_stats["404"] + groq_stats["failed"] >= counts["groq"]
assert mr.provider_stats("mistral")["requests"] == counts["mistral"]
assert mr.provider_stats("mistral")["successful"] == counts["mistral"]
ok("no lost counter updates: requests/successful match the real call log")

# ══════════════════════════════════════════════════════════════════════════
print("\n2. Once latched, later threads never touch groq at all")
before = Counter(calls)["groq"]
with patch.dict(os.environ, ENV), \
     patch.object(mr, "_call_groq", stub("groq", GROQ_404)), \
     patch.object(mr, "_call_openrouter", stub("openrouter", "429 Too Many Requests")), \
     patch.object(mr, "_call_mistral", stub("mistral")), \
     patch.object(mr, "_call_gemini", stub("gemini")), \
     patch.object(mr, "wait_for_quota"), \
     patch.object(mr, "get_daily_claude_spend", return_value=0.0), \
     patch.object(mr, "get_monthly_claude_spend", return_value=0.0):
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: mr.call_model("first_pass_filter", "p"), range(40)))

assert Counter(calls)["groq"] == before, "groq was re-attempted after latching"
ok("40 further concurrent operations, 0 additional groq calls")

# ══════════════════════════════════════════════════════════════════════════
print("\n3. reset_provider_stats() is atomic and re-arms groq for a new scan")
mr.reset_provider_stats()
assert mr.provider_stats() == {}
calls.clear()
with patch.dict(os.environ, ENV), \
     patch.object(mr, "_call_groq", stub("groq")), \
     patch.object(mr, "_call_mistral", stub("mistral")), \
     patch.object(mr, "wait_for_quota"), \
     patch.object(mr, "get_daily_claude_spend", return_value=0.0), \
     patch.object(mr, "get_monthly_claude_spend", return_value=0.0):
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: mr.call_model("first_pass_filter", "p"), range(40)))
    assert Counter(calls)["groq"] == 40, Counter(calls)
    assert mr.provider_state("groq") == mr.AVAILABLE
    assert mr.provider_stats("groq")["requests"] == 40
    assert mr.provider_stats("groq")["config_error"] is False
ok("new scan: 40 concurrent successes, groq AVAILABLE, counters exact")

print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
