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

import checker

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
assert checker._as_object({"a": 1}, "t") == {"a": 1}
assert checker._as_object([1, 2], "t") == {}
assert checker._as_object(None, "t") == {}
assert checker._as_object("str", "t") == {}
assert checker._as_object(7, "t") == {}
assert checker._as_object(True, "t") == {}
ok("dict passes through; list/None/str/int/bool all yield {}")

# It must say something when it discards a reply — silence would hide this
# recurring, provider-specific behaviour.
import io as _io
from contextlib import redirect_stdout
buf = _io.StringIO()
with redirect_stdout(buf):
    checker._as_object([{"verdict": "scam"}], "scam_detection")
    checker._as_object(None, "scam_detection")
printed = buf.getvalue()
assert "scam_detection" in printed and "list" in printed, printed
assert printed.count("⚠️") == 1, "None is 'no JSON found', not a shape error"
ok("a discarded reply is logged with its type; a plain None is not")

print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
