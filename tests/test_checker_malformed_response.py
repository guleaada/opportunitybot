#!/usr/bin/env python3
"""check_legitimacy() must survive a non-object JSON reply.

A production scan died at checker.py:284 with

    AttributeError: 'list' object has no attribute 'get'

after Groq was rate-limited and the OpenRouter fallback answered with a JSON
array. extract_json() returns the first balanced {...} OR [...], so a list is
a shape the parser can legitimately produce — the caller has to cope with it.

A malformed reply must NOT be coerced into a verdict. It resolves to
verdict "unknown", which the credibility ladder maps to NEEDS_VERIFICATION —
the project's existing "could not verify" outcome.

Run:  python tests/test_checker_malformed_response.py
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest import TestCase
from analysis_response import AnalysisResponseError
import checker
from model_router import as_object

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  ✅ {msg}")


def reply(content):
    return {"content": content, "model_used": "stub", "task_type": "scam_detection",
            "tokens_used": {"input": 1, "output": 1}, "cost_usd": 0.0,
            "fell_back": True}


def legitimacy(content, url="https://example.org/fellowship"):
    with patch.object(checker, "call_model", return_value=reply(content)):
        return checker.check_legitimacy("Some page text about a fellowship.", url)


# ══════════════════════════════════════════════════════════════════════════
print("\n1. The exact production failure: a JSON array reply")
# What OpenRouter actually returned in shape: a top-level array.
out = legitimacy('[{"verdict": "legitimate", "confidence": 0.9, '
                 '"reasoning": "r", "red_flags": []}]')
assert isinstance(out, dict), out
assert out["verdict"] == "unknown", out["verdict"]
assert out["credibility_status"] == checker.NEEDS_VERIFICATION, out
ok("array reply → no AttributeError, verdict 'unknown', NEEDS_VERIFICATION")

# The array's contents must NOT be unwrapped into a verdict.
assert out["verdict"] != "legitimate", "a wrapped object must not be trusted"
assert out["confidence"] is None, out["confidence"]
ok("the array's inner object is discarded, not promoted — no invented verdict")

# ══════════════════════════════════════════════════════════════════════════
print("\n2. Every non-object shape is safe")
for label, content in (
    ("array of objects", '[{"verdict": "scam"}]'),
    ("array of strings", '["legitimate", "scam"]'),
    ("empty array",      "[]"),
    ("nested arrays",    "[[1, 2], [3]]"),
    ("bare string",      "just prose, no json at all"),
    ("empty content",    ""),
    ("null",             "null"),
    ("fenced array",     '```json\n[{"verdict": "scam"}]\n```'),
):
    out = legitimacy(content)
    assert isinstance(out, dict), (label, out)
    assert out["verdict"] == "unknown", (label, out["verdict"])
    assert out["credibility_status"] == checker.NEEDS_VERIFICATION, (label, out)
ok("8 malformed shapes → all 'unknown' / NEEDS_VERIFICATION, none raised")

# The full return contract still holds on the malformed path.
out = legitimacy("[]")
for key in ("verdict", "credibility_status", "source_tier", "confidence",
            "reasoning", "red_flags", "model_used", "cost_usd"):
    assert key in out, (key, out)
assert isinstance(out["red_flags"], list), out["red_flags"]
ok("return shape unchanged on the malformed path (all 8 keys present)")

# ══════════════════════════════════════════════════════════════════════════
print("\n3. The object path is completely unchanged")
out = legitimacy('{"verdict": "legitimate", "confidence": 0.95, '
                 '"reasoning": "Official DAAD program.", "red_flags": []}')
assert out["verdict"] == "legitimate", out
assert out["confidence"] == 0.95
assert out["reasoning"] == "Official DAAD program."
assert out["red_flags"] == []
ok("a well-formed object still yields its verdict, confidence and reasoning")

for verdict in ("legitimate", "scam", "suspicious", "unknown"):
    out = legitimacy('{"verdict": "%s", "confidence": 0.8}' % verdict)
    assert out["verdict"] == verdict, (verdict, out)
ok("all four documented verdicts pass through untouched")

# An unrecognised verdict string is still normalised to unknown, as before.
out = legitimacy('{"verdict": "definitely-fine", "confidence": 0.9}')
assert out["verdict"] == "unknown", out
ok("an out-of-contract verdict string still normalises to 'unknown'")

# ══════════════════════════════════════════════════════════════════════════
print("\n4. _as_object() in isolation")
assert as_object({"a": 1}, "t") == {"a": 1}
assert as_object([1, 2], "t") == {}
assert as_object(None, "t") == {}
assert as_object("str", "t") == {}
assert as_object(7, "t") == {}
assert as_object(True, "t") == {}
ok("dict passes through; list/None/str/int/bool all yield {}")

# It must say something when it discards a reply — silence would hide this
# recurring, provider-specific behaviour.
import io as _io
from contextlib import redirect_stdout
buf = _io.StringIO()
with redirect_stdout(buf):
    as_object([{"verdict": "scam"}], "scam_detection")
    as_object(None, "scam_detection")
printed = buf.getvalue()
assert "scam_detection" in printed and "list" in printed, printed
assert printed.count("⚠️") == 1, "None is 'no JSON found', not a shape error"
ok("a discarded reply is logged with its type; a plain None is not")

# ══════════════════════════════════════════════════════════════════════════
print("\n5. Every other extract_json call site survives an array reply")
import scorer
import signals
import tools

ARRAY = '[{"verdict": "scam", "keep": false, "overall_score": 9.9}]'


def _stub(module, content):
    return patch.object(module, "call_model", return_value=reply(content))


# check_deadline — must not invent a deadline
with _stub(checker, ARRAY):
    out = checker.check_deadline("Applications close soon.")
assert out["status"] == "unknown" and out["deadline"] is None, out
ok("check_deadline  → status 'unknown', no deadline invented")

# Decision failures are retryable, never zero scores or eligibility verdicts.
with _stub(checker, ARRAY), TestCase().assertRaises(AnalysisResponseError):
    checker.check_eligibility("Some program text.", {})
ok("check_eligibility → retryable response error")
with _stub(scorer, ARRAY), TestCase().assertRaises(AnalysisResponseError):
    scorer.score_opportunity({"raw_text":"t", "url":"https://e/x"}, {})
ok("score_opportunity → retryable response error")

# _classify_with_cheap_model — an unusable reply must not become a yes
with _stub(signals, ARRAY):
    out = signals._classify_with_cheap_model("text", "title", {"weak": [],
                                                               "strong": []})
assert out["is_opportunity"] is False, out
ok("classify_opportunity → is_opportunity False, not a yes")

# extract_documents / estimate_complexity — documented defaults
with _stub(tools, ARRAY):
    out = tools.extract_documents("text")
assert out["notes"] == "extraction failed" and out["documents"] == [], out
ok("extract_documents → documented 'extraction failed' default")

with _stub(tools, ARRAY):
    out = tools.estimate_complexity("text")
assert out["notes"] == "estimation failed", out
assert out["difficulty"] == "unknown" and out["odds"] == "unknown", out
ok("estimate_complexity → documented 'estimation failed' default")

# Not one of the six raises AttributeError on any non-object shape.
for shape in ("[]", "[1,2]", "null", '"str"', "42", "no json here"):
    with _stub(checker, shape):
        checker.check_deadline("t")
        with TestCase().assertRaises(AnalysisResponseError):
            checker.check_eligibility("t", {})
        checker.check_legitimacy("t", "https://e/x")
    with _stub(signals, shape):
        signals._classify_with_cheap_model("t", "", {"weak": [], "strong": []})
    with _stub(tools, shape):
        tools.extract_documents("t")
        tools.estimate_complexity("t")
    with _stub(scorer, shape), patch.object(scorer, "profile_summary",
                                            return_value="p"):
        with TestCase().assertRaises(AnalysisResponseError):
            scorer.score_opportunity({"raw_text": "t", "url": "https://e/x"}, {})
ok("6 shapes x 7 call sites = 42 combinations, zero AttributeErrors")


print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
