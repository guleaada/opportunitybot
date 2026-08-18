#!/usr/bin/env python3
"""Regression tests reproducing production run 79's Groq failure.

Run 79 logged `Groq requests 50, ok 0, 404 50` — the same
"The model `…` does not exist or you do not have access to it." error
repeated across clean_html, first_pass_filter, classify_opportunity,
scam_detection and deep_eligibility, for every candidate.

These tests replay that exact operation sequence and assert the latch holds
across operation boundaries and across candidates: Groq is attempted once,
then skipped for the remainder of the scan, and the next available provider
(Mistral, since OpenRouter is rate-limited) serves everything after it.

Run:  python tests/test_provider_latch_e2e.py
"""
import os
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_router as mr

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  ✅ {msg}")


# The exact Groq error from run 79, in both shapes an SDK might raise it:
# carrying an HTTP status, and carrying only the message text.
GROQ_404_MESSAGE = ("Error code: 404 - {'error': {'message': 'The model "
                    "`some-retired-model` does not exist or you do not "
                    "have access to it.', 'type': 'invalid_request_error'}}")


class _Response:
    status_code = 404


class GroqNotFoundError(Exception):
    """Stand-in for groq.NotFoundError (status attribute + message)."""
    status_code = 404
    response = _Response()


GROQ_404_WITH_STATUS = GroqNotFoundError(GROQ_404_MESSAGE)
GROQ_404_TEXT_ONLY = RuntimeError(GROQ_404_MESSAGE)

# The five operations run 79 showed Groq being attempted on, plus scoring.
# scam_detection / deep_eligibility / final_scoring are Claude-routed and
# downgrade into the free chain at budget 0, so this also covers that path.
OPERATIONS = ["clean_html", "first_pass_filter", "classify_opportunity",
              "scam_detection", "deep_eligibility", "final_scoring"]

OK_RESPONSE = {"content": "x", "model_used": "m", "task_type": "t",
               "tokens_used": {"input": 1, "output": 1}, "cost_usd": 0.0,
               "fell_back": False}

# Run 79's environment: every free provider keyed, no Claude key, budget 0.
ENV = {"GROQ_API_KEY": "g", "OPENROUTER_API_KEY": "o", "MISTRAL_API_KEY": "m",
       "GEMINI_API_KEY": "ge", "DAILY_CLAUDE_BUDGET_USD": "0",
       "ANTHROPIC_API_KEY": ""}


class Scan:
    """One scan: module state is reset once at the start, as main.py does."""

    def __init__(self, groq_error, openrouter_error=None):
        self.calls = []                      # (provider, task_type)
        self.groq_error = groq_error
        self.openrouter_error = openrouter_error

    def _stub(self, name, error=None):
        def f(prompt, system, max_tokens, temperature, task_type, *a, **k):
            self.calls.append((name, task_type))
            if error is not None:
                raise error
            return dict(OK_RESPONSE)
        return f

    def __enter__(self):
        mr.reset_provider_stats()
        self._patches = [
            patch.dict(os.environ, ENV),
            patch.object(mr, "_call_groq", self._stub("groq", self.groq_error)),
            patch.object(mr, "_call_openrouter",
                         self._stub("openrouter", self.openrouter_error)),
            patch.object(mr, "_call_mistral", self._stub("mistral")),
            patch.object(mr, "_call_gemini", self._stub("gemini")),
            patch.object(mr, "wait_for_quota"),
            patch.object(mr, "get_daily_claude_spend", return_value=0.0),
            patch.object(mr, "get_monthly_claude_spend", return_value=0.0),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False

    def candidate(self, operations=OPERATIONS):
        """Run one candidate's full sequence of model operations."""
        for task in operations:
            try:
                mr.call_model(task, "prompt")
            except mr.ProvidersUnavailable:
                pass

    def by_provider(self):
        return Counter(p for p, _ in self.calls)

    def tasks_for(self, provider):
        return [t for p, t in self.calls if p == provider]


# ══════════════════════════════════════════════════════════════════════════
# 7. Candidate 1 hits the Groq 404; candidate 2 must skip Groq entirely
# ══════════════════════════════════════════════════════════════════════════
print("\n7. Groq 404 on candidate 1 → candidate 2 never touches Groq")
OPENROUTER_429 = RuntimeError("429 Client Error: Too Many Requests")

with Scan(GROQ_404_WITH_STATUS, OPENROUTER_429) as scan:
    scan.candidate()                                   # candidate 1
    first = scan.by_provider()
    assert first["groq"] == 1, f"groq attempted {first['groq']}x on candidate 1"
    assert mr.provider_stats("groq")["config_error"] is True
    assert mr.provider_state("groq") == mr.FAILED
    ok(f"candidate 1: groq attempted once, latched FAILED "
       f"(run 79 attempted it {6} times per candidate)")

    for _ in range(9):                                 # candidates 2-10
        scan.candidate()

counts = scan.by_provider()
assert counts["groq"] == 1, f"groq called {counts['groq']}x across 10 candidates"
assert scan.tasks_for("groq") == ["clean_html"], scan.tasks_for("groq")
ok(f"10 candidates x 6 operations = 60 slots → groq called exactly "
   f"{counts['groq']}x (was 60)")

# ══════════════════════════════════════════════════════════════════════════
# 8. The latch crosses operation boundaries within a single candidate
# ══════════════════════════════════════════════════════════════════════════
print("\n8. first_pass_filter 404 → scam_detection and deep_eligibility skip Groq")
with Scan(GROQ_404_WITH_STATUS, OPENROUTER_429) as scan:
    mr.call_model("first_pass_filter", "p")
    assert scan.tasks_for("groq") == ["first_pass_filter"]
    assert mr.provider_state("groq") == mr.FAILED

    for task in ("scam_detection", "deep_eligibility", "classify_opportunity",
                 "clean_html", "final_scoring", "extract_text", "translate",
                 "generate_cover_letter", "summarize_opportunity"):
        mr.call_model(task, "p")
        assert "groq" not in [p for p, t in scan.calls if t == task], \
            f"groq was attempted for {task} after latching"
ok("9 further operation types — including Claude-routed and Groq-routed "
   "tasks — all skip groq")

# ══════════════════════════════════════════════════════════════════════════
# 9. Mistral is the next provider that actually serves the work
# ══════════════════════════════════════════════════════════════════════════
print("\n9. Mistral takes over as the next available provider")
assert mr.provider_order() == ["groq", "openrouter", "mistral", "gemini"], \
    mr.provider_order()
# provider_state() reads the environment for credentials, so these assertions
# must run inside the scan (where the keys are patched in), not after it.
with Scan(GROQ_404_WITH_STATUS, OPENROUTER_429) as scan:
    scan.candidate()
    scan.candidate()
    counts = scan.by_provider()

    assert counts["groq"] == 1, counts
    assert counts["openrouter"] == 1, counts      # 429 → RATE_LIMITED, skipped
    assert counts["mistral"] == len(OPERATIONS) * 2, counts
    assert counts["gemini"] == 0, "gemini must not be reached while mistral works"
    assert mr.provider_state("mistral") == mr.AVAILABLE
    assert mr.provider_state("openrouter") == mr.RATE_LIMITED
    assert mr.provider_state("groq") == mr.FAILED
    ok(f"groq 1 (FAILED), openrouter 1 (RATE_LIMITED), "
       f"mistral {counts['mistral']} (AVAILABLE), gemini 0 — order preserved")

    # The two breakers stay distinct even though both skip the provider.
    assert mr.provider_stats("groq")["404"] == 1
    assert mr.provider_stats("groq")["429"] == 0
    assert mr.provider_stats("openrouter")["429"] == 1
    assert mr.provider_stats("openrouter")["config_error"] is False
    ok("404 latch and 429 breaker remain distinct states")

# ══════════════════════════════════════════════════════════════════════════
# 5. Transient errors must NOT latch — 429/500/502/503/timeout
# ══════════════════════════════════════════════════════════════════════════
print("\n5. Transient failures stay retryable")
TRANSIENT = [
    ("429", RuntimeError("429 Too Many Requests")),
    ("500", RuntimeError("500 Internal Server Error")),
    ("502", RuntimeError("502 Bad Gateway")),
    ("503", RuntimeError("503 Service Unavailable")),
    ("timeout", RuntimeError("Read timed out. (read timeout=25)")),
    ("conn reset", RuntimeError("Connection aborted, ConnectionResetError")),
]
for label, err in TRANSIENT:
    assert mr._is_model_not_found(err) is False, \
        f"{label} was misclassified as a permanent model error"
ok(f"{len(TRANSIENT)} transient errors classified as retryable, not latched")

# 500/502/503/timeout leave the provider retryable on the next candidate.
for label, err in TRANSIENT[1:]:
    with Scan(err) as scan:
        scan.candidate(["first_pass_filter"])
        scan.candidate(["first_pass_filter"])
        scan.candidate(["first_pass_filter"])
        assert scan.by_provider()["groq"] == 3, (label, scan.by_provider())
        assert mr.provider_stats("groq")["config_error"] is False, label
ok("500/502/503/timeout/reset → groq retried on every candidate (not latched)")

# 429 latches as RATE_LIMITED, which is a different state with the same skip.
with Scan(RuntimeError("429 Too Many Requests")) as scan:
    scan.candidate(["first_pass_filter"])
    scan.candidate(["first_pass_filter"])
    assert scan.by_provider()["groq"] == 1, scan.by_provider()
    assert mr.provider_state("groq") == mr.RATE_LIMITED
    assert mr.provider_stats("groq")["config_error"] is False
ok("429 → RATE_LIMITED (not config_error), still skipped for the scan")

# ══════════════════════════════════════════════════════════════════════════
# The production error text alone is enough, even with no status attribute
# ══════════════════════════════════════════════════════════════════════════
print("\n+ Classification of run 79's exact error string")
assert mr._is_model_not_found(GROQ_404_WITH_STATUS) is True
assert mr._is_model_not_found(GROQ_404_TEXT_ONLY) is True, \
    "the message text alone must be enough — do not rely on the SDK's shape"
assert mr._is_rate_limit(GROQ_404_WITH_STATUS) is False, \
    "a 404 must not be read as a rate limit (that would mask it)"
with Scan(GROQ_404_TEXT_ONLY, OPENROUTER_429) as scan:
    scan.candidate()
    scan.candidate()
    assert scan.by_provider()["groq"] == 1, scan.by_provider()
ok("run 79's error latches via status_code AND via message text")

# ══════════════════════════════════════════════════════════════════════════
# The latch is per-scan, not permanent: a new scan gives Groq another chance
# ══════════════════════════════════════════════════════════════════════════
print("\n+ The latch resets between scans")
with Scan(GROQ_404_WITH_STATUS, OPENROUTER_429) as scan:
    scan.candidate(["first_pass_filter"])
    assert mr.provider_state("groq") == mr.FAILED
with Scan(None) as scan:                    # next scan, model name fixed
    scan.candidate(["first_pass_filter"])
    assert scan.by_provider()["groq"] == 1
    assert mr.provider_state("groq") == mr.AVAILABLE
ok("reset_provider_stats() at scan start clears the latch — not sticky state")

print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
