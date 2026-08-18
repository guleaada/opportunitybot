#!/usr/bin/env python3
"""Regression tests for the two call-reduction fixes.

  1. A PERMANENT provider fault (404 / bad model name) latches: the provider
     is skipped for the rest of the scan instead of failing identically once
     per candidate. Transient failures stay retryable; 429 keeps using
     RATE_LIMITED; a missing key keeps using DISABLED.
  2. clean_html is local — it makes no model call and no HTTP request.

Also measures model calls per candidate through the real pipeline, so the
call-amplification number is verified rather than asserted from memory.

Run:  python tests/test_call_reduction.py
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import model_router as mr
import tools

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  ✅ {msg}")


OK_RESPONSE = {"content": "x", "model_used": "m", "task_type": "t",
               "tokens_used": {"input": 1, "output": 1}, "cost_usd": 0.0,
               "fell_back": False}
KEYS = {"GROQ_API_KEY": "g", "OPENROUTER_API_KEY": "o", "MISTRAL_API_KEY": "m",
        "GEMINI_API_KEY": "ge", "DAILY_CLAUDE_BUDGET_USD": "0"}


class Chain:
    """Drive call_model with every provider stubbed; record who was called."""

    def __init__(self, env=None, **behaviour):
        self.calls = []
        self.behaviour = behaviour
        self.env = {**KEYS, **(env or {})}

    def _stub(self, name):
        def f(*a, **k):
            self.calls.append(name)
            b = self.behaviour.get(name)
            if isinstance(b, Exception):
                raise b
            if callable(b):
                return b()
            return dict(OK_RESPONSE)
        return f

    def __enter__(self):
        mr.reset_provider_stats()          # a fresh scan
        self._patches = [
            patch.dict(os.environ, self.env),
            patch.object(mr, "_call_groq", side_effect=self._stub("groq")),
            patch.object(mr, "_call_openrouter", side_effect=self._stub("openrouter")),
            patch.object(mr, "_call_mistral", side_effect=self._stub("mistral")),
            patch.object(mr, "_call_gemini", side_effect=self._stub("gemini")),
            patch.object(mr, "_call_claude", side_effect=self._stub("claude")),
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

    def candidate(self, task="first_pass_filter"):
        """One candidate's worth of work; returns the result or the exception."""
        try:
            return mr.call_model(task, "prompt")
        except Exception as e:
            return e


# ══════════════════════════════════════════════════════════════════════════
# 1. Gemini 404 on candidate 1 latches the provider as FAILED
# ══════════════════════════════════════════════════════════════════════════
print("\n1. Gemini 404 latches the provider")
GEMINI_404 = RuntimeError(
    "404 This model models/gemini-1.5-flash is no longer available to new "
    "users. Please update your code to use a newer model.")

with Chain(groq=RuntimeError("groq down"),
           openrouter=RuntimeError("openrouter down"),
           mistral=RuntimeError("mistral down"),
           gemini=GEMINI_404) as c:
    res1 = c.candidate()
    assert isinstance(res1, mr.ProvidersUnavailable), type(res1)
    assert c.calls == ["groq", "openrouter", "mistral", "gemini"], c.calls
    s = mr.provider_stats("gemini")
    assert s["404"] == 1, s
    assert s["config_error"] is True, s
    assert mr.provider_state("gemini") == mr.FAILED, mr.provider_state("gemini")
    ok("candidate 1: gemini 404 → 404 counted, config_error latched, state FAILED")

    # ── 2. Candidates 2..N must NOT call gemini again ──────────────────────
    before = len(c.calls)
    for _ in range(4):
        c.candidate()
    later = c.calls[before:]
    assert "gemini" not in later, f"gemini was retried after latching: {later}"
    assert mr.provider_stats("gemini")["requests"] == 1, mr.provider_stats("gemini")
    ok(f"candidates 2-5: gemini never called again (1 request total, not 5)")

    # The other providers failed transiently, so they ARE retried.
    assert later.count("groq") == 4, later
    ok("transient-failure providers are still retried on every candidate")

# ══════════════════════════════════════════════════════════════════════════
# 3. A transient failure does NOT permanently disable the provider
# ══════════════════════════════════════════════════════════════════════════
print("\n3. Transient failures stay retryable")
_flaky = {"n": 0}


def flaky():
    _flaky["n"] += 1
    if _flaky["n"] == 1:
        raise RuntimeError("503 Service Unavailable — upstream hiccup")
    return dict(OK_RESPONSE)


with Chain(groq=flaky) as c:
    r1 = c.candidate()                       # fails over to openrouter
    assert mr.provider_stats("groq")["config_error"] is False
    assert mr._provider_available("groq") is True, "transient error latched!"
    r2 = c.candidate()                       # groq gets another chance
    assert c.calls == ["groq", "openrouter", "groq"], c.calls
    assert mr.provider_state("groq") == mr.AVAILABLE, mr.provider_state("groq")
    ok("503 → retried next candidate → recovers to AVAILABLE")

# "service unavailable" and "not found" on a host must not read as a bad model
assert mr._is_model_not_found(RuntimeError("503 Service Unavailable")) is False
assert mr._is_model_not_found(RuntimeError("host not found: api.example")) is False
assert mr._is_model_not_found(GEMINI_404) is True
assert mr._is_model_not_found(RuntimeError("model_not_found")) is True
assert mr._is_model_not_found(RuntimeError("404 models/x is not found")) is True
ok("permanent vs transient classification: transient errors not misread")

# ══════════════════════════════════════════════════════════════════════════
# 4. 429 still produces RATE_LIMITED — the states stay distinct
# ══════════════════════════════════════════════════════════════════════════
print("\n4. 429 → RATE_LIMITED, and the three states stay distinct")
with Chain(groq=RuntimeError("429 Too Many Requests"),
           gemini=GEMINI_404,
           openrouter=RuntimeError("503 transient"),
           env={"MISTRAL_API_KEY": ""}) as c:
    c.candidate()
    assert mr.provider_state("groq") == mr.RATE_LIMITED
    assert mr.provider_state("gemini") == mr.FAILED
    assert mr.provider_state("mistral") == mr.DISABLED
    assert mr.provider_stats("groq")["429"] == 1
    assert mr.provider_stats("groq")["config_error"] is False, \
        "a 429 must not be latched as a configuration error"
    assert len({mr.RATE_LIMITED, mr.FAILED, mr.DISABLED}) == 3
    ok("groq=RATE_LIMITED  gemini=FAILED  mistral=DISABLED — all distinct")

    before = len(c.calls)
    c.candidate()
    later = c.calls[before:]
    assert "groq" not in later and "gemini" not in later, later
    assert "openrouter" in later, "transient provider should still be tried"
    ok("next candidate skips both the rate-limited and the latched provider")

# ══════════════════════════════════════════════════════════════════════════
# 9. The existing free-provider fallback still works
# ══════════════════════════════════════════════════════════════════════════
print("\n9. Provider fallback preserved")
assert mr.provider_order() == ["groq", "openrouter", "mistral", "gemini"], \
    mr.provider_order()
with Chain() as c:
    c.candidate()
    assert c.calls == ["groq"], c.calls
ok("healthy run uses groq only")

with Chain(groq=RuntimeError("groq down")) as c:
    r = c.candidate()
    assert c.calls == ["groq", "openrouter"], c.calls
    assert r["fell_back"] is True
ok("groq fails → openrouter serves it; gemini untouched")

with Chain(groq=RuntimeError("down"), openrouter=RuntimeError("down"),
           mistral=RuntimeError("down")) as c:
    c.candidate()
    assert c.calls == ["groq", "openrouter", "mistral", "gemini"], c.calls
ok("full chain walked in order; gemini is the last resort")

with Chain(groq=RuntimeError("down"), openrouter=RuntimeError("down"),
           mistral=RuntimeError("down"), gemini=GEMINI_404) as c:
    r = c.candidate()
    assert isinstance(r, mr.ProvidersUnavailable)
    assert "claude" not in c.calls, "paid Claude called with budget 0!"
ok("all free providers down + budget 0 → ProvidersUnavailable, no paid call")

# ══════════════════════════════════════════════════════════════════════════
# 5-8. clean_html: no model call, no HTTP, correct output, never crashes
# ══════════════════════════════════════════════════════════════════════════
print("\n5. clean_html makes zero model/API calls")


def _boom(*a, **k):
    raise AssertionError("clean_html made a network/model call!")


HTML = """
<html><head><title>T</title>
<style>.x{color:red}</style>
<script>var t = 1 < 2 && 3 > 2;</script></head>
<body><nav>Home | About</nav>
<h1>DAAD&nbsp;EPOS Scholarship</h1>
<p>Fully funded for professionals &amp; researchers.</p>
<p>Deadline: 31&nbsp;August&nbsp;2026</p>
<footer>&copy; 2026</footer></body></html>
"""

with patch.object(tools, "call_model", side_effect=_boom), \
     patch("requests.get", side_effect=_boom), \
     patch("requests.post", side_effect=_boom):
    out = tools.clean_html(HTML)
    tools.clean_html("plain text, already extracted")
    tools.clean_html("")
ok("clean_html ran with call_model/requests booby-trapped — zero calls")

print("\n6. clean_html strips HTML correctly")
assert "<" not in out and ">" not in out, out
assert "var t" not in out and "color:red" not in out, out
assert "DAAD EPOS Scholarship" in out, out          # &nbsp; → space
assert "professionals & researchers" in out, out    # &amp;  → &
assert "31 August 2026" in out, out
assert "Home | About" not in out, "nav should be dropped"
assert " " not in out and "\t" not in out, "whitespace not normalized"
assert "\n\n" not in out, "blank lines should be collapsed"
ok("tags/scripts/styles/nav removed, entities decoded, whitespace normalized")

# Semantics preserved: every substantive word survives.
for word in ("DAAD", "EPOS", "Scholarship", "Fully", "funded", "Deadline"):
    assert word in out, f"{word} lost during cleaning"
ok("no substantive content lost")

print("\n7. Malformed HTML does not crash")
for bad in ("<div><p>unclosed <b>tags <<>> &notanentity; </p>",
            "<a href='x>broken quote</a>",
            "not xml <> < > </>",
            "<script>never closed",
            "<<<<>>>>",
            "<p>text</p" ):
    r = tools.clean_html(bad)
    assert isinstance(r, str), (bad, r)
ok("6 malformed inputs handled, all returned str")

print("\n8. Empty / odd input does not crash")
assert tools.clean_html("") == ""
assert tools.clean_html(None) == ""
assert tools.clean_html("   \n\t  ") == ""
assert tools.clean_html("<div>   </div>") == ""
ok("empty, None and whitespace-only inputs return '' without raising")

big = ("<p>Fully funded fellowship for Ethiopian professionals. </p>" * 20000)
r = tools.clean_html(big)
assert isinstance(r, str) and len(r) <= tools._MAX_TEXT + 100, len(r)
assert "Fully funded fellowship" in r
ok(f"1.2 MB input handled and bounded to {len(r)} chars")

# ══════════════════════════════════════════════════════════════════════════
# 10. All providers unavailable → the candidate is PRESERVED, not discarded
# ══════════════════════════════════════════════════════════════════════════
print("\n10. Candidate preserved when every provider is unavailable")
import search
import main as app

saved, watchlisted = [], []
with patch.object(app.tools, "fetch_url", return_value={
        "url": "https://example.org/x", "html": "<p>Fellowship</p>",
        "text": "Fully funded fellowship for professionals. Deadline 2026.",
        "status": 200, "cached": False, "error": None}), \
     patch.object(app.tools, "save_opportunity", side_effect=saved.append), \
     patch.object(app.tools, "add_to_watchlist", side_effect=watchlisted.append), \
     patch.object(app.tools, "first_pass_filter",
                  side_effect=mr.ProvidersUnavailable(
                      "all configured providers unavailable")):
    stats = Counter()
    r = search.SearchResult(title="Some Fellowship",
                            url="https://example.org/x", source="example.org")
    out10 = app.analyze_one(r, stats)

assert out10 is None, out10
assert stats["analysis_unavailable"] == 1, dict(stats)
assert watchlisted and "ANALYSIS_UNAVAILABLE" in watchlisted[0]["reason"], watchlisted
assert not saved, "candidate must NOT be marked seen — it has to be retried"
ok("ANALYSIS_UNAVAILABLE: watchlisted, not marked seen, never silently dropped")

# ══════════════════════════════════════════════════════════════════════════
# 11. Measured call amplification — every model call counted at the router
# ══════════════════════════════════════════════════════════════════════════
print("\n11. Model calls per candidate (measured, not estimated)")
import checker
import scorer
import signals

COUNTS = Counter()
PAGE = ("DAAD EPOS Scholarship. Fully funded masters for working professionals "
        "from developing countries, open to Ethiopian nationals. Application "
        "deadline 31 December 2026. No application fee.")
BLAND = "A blog post about the weather in Berlin. Nothing of interest here."
CANNED = {
    "first_pass_filter": {"keep": True, "reason": "eligible",
                          "guessed_country": "Germany",
                          "guessed_funding": "fully_funded"},
    "scam_detection": {"verdict": "legitimate", "confidence": 0.95,
                       "reasoning": "official", "red_flags": []},
    "deep_eligibility": {"overall": "eligible", "reasoning": "meets rules",
                         "blocking_issues": [], "addressable_gaps": []},
    "extract_document_requirements": {"documents": ["CV"],
                                      "english_test_required": "Duolingo",
                                      "references_required": 2,
                                      "transcripts_required": True, "notes": ""},
    "estimate_complexity": {"estimated_hours": 25, "difficulty": "medium",
                            "odds": "moderate", "notes": ""},
    "final_scoring": {"overall_score": 9.2, "breakdown": {}, "funding":
                      "fully_funded", "reasoning": "good", "recommendation": "apply"},
    "classify_opportunity": {"is_opportunity": False, "reason": "",
                             "confidence": 0.9},
}
DEADLINE = {"deadline_iso": "2026-12-31", "deadline_raw": "31 December 2026",
            "is_explicitly_closed": False, "found": True}


def counting_call_model(task_type, prompt, system=None, tools=None,
                        max_tokens=2048, temperature=0.3):
    COUNTS[task_type] += 1
    if task_type == "first_pass_filter" and "APPLICATION deadline" in prompt:
        payload = DEADLINE
    else:
        payload = CANNED.get(task_type, {"keep": True})
    return {"content": json.dumps(payload), "model_used": "stub",
            "task_type": task_type, "tokens_used": {"input": 1, "output": 1},
            "cost_usd": 0.0, "fell_back": False}


def measure(page, filt_keep, url):
    CANNED["first_pass_filter"] = (
        {"keep": True, "reason": "eligible", "guessed_country": "Germany",
         "guessed_funding": "fully_funded"} if filt_keep else
        {"keep": False, "reason": "not relevant", "guessed_country": "",
         "guessed_funding": ""})
    COUNTS.clear()
    stubs = [patch.object(m, "call_model", counting_call_model)
             for m in (mr, checker, scorer, signals, tools)
             if hasattr(m, "call_model")]
    stubs += [
        patch.object(app.tools, "fetch_url", return_value={
            "url": url, "html": f"<html><body><p>{page}</p></body></html>",
            "text": page, "status": 200, "cached": False, "error": None}),
        patch.object(app.tools, "save_opportunity", lambda d: None),
        patch.object(app.tools, "add_to_watchlist", lambda d: None),
        patch.object(app.tools, "add_to_calendar", lambda d: {"added": False}),
    ]
    for p in stubs:
        p.start()
    try:
        app.analyze_one(search.SearchResult(title="DAAD EPOS Scholarship",
                                            url=url, source="daad.de"),
                        Counter())
    finally:
        for p in reversed(stubs):
            p.stop()
    return sum(COUNTS.values())


full = measure(PAGE, True, "https://www.daad.de/epos-full")
assert COUNTS["clean_html"] == 0, "clean_html still makes a model call!"
early = measure(BLAND, False, "https://www.daad.de/epos-early")
assert COUNTS["clean_html"] == 0, "clean_html still makes a model call!"

# Before this change: 8 on the full path, 2 on the early-exit path — one
# clean_html call per candidate, spent before any gate could drop it.
assert full == 7, f"expected 7 model calls on the full path, measured {full}"
assert early == 1, f"expected 1 model call on the early-exit path, got {early}"
ok(f"full path 8 → {full} calls/candidate; "
   f"early exit 2 → {early} call/candidate; clean_html 0")

print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
