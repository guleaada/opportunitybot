#!/usr/bin/env python3
"""_google_error_reason() must name the argument Google is rejecting.

Runs 83-85 logged `badRequest: Request contains an invalid argument.` for
every Google Custom Search query, with no indication of WHICH argument —
because errors[].location / locationType / domain and error.status were
parsed and then thrown away. This is the field that identifies the culprit
(typically `cx`), so it has to reach the log.

Diagnostics only: nothing here changes the request or the search behaviour.

Run:  python tests/test_google_error_reason.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import search

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  ✅ {msg}")


class FakeResponse:
    """Minimal stand-in for requests.Response — only .json() is used."""

    def __init__(self, payload, raises=False):
        self._payload = payload
        self._raises = raises

    def json(self):
        if self._raises:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


# The shape Google's Custom Search JSON API returns for a bad `cx`.
GOOGLE_400 = {
    "error": {
        "code": 400,
        "message": "Request contains an invalid argument.",
        "errors": [{
            "message": "Request contains an invalid argument.",
            "domain": "global",
            "reason": "badRequest",
            "location": "cx",
            "locationType": "parameter",
        }],
        "status": "INVALID_ARGUMENT",
    }
}

# ══════════════════════════════════════════════════════════════════════════
print("\n1. All six fields reach the diagnostic")
out = search._google_error_reason(FakeResponse(GOOGLE_400))
print(f"      → {out}")
for field, value in (("reason", "badRequest"),
                     ("location", "cx"),
                     ("locationType", "parameter"),
                     ("domain", "global"),
                     ("status", "INVALID_ARGUMENT"),
                     ("message", "Request contains an invalid argument.")):
    assert value in out, f"{field} ({value!r}) missing from: {out}"
ok("reason, location, locationType, domain, status and message all present")

# The whole point: the parameter is identifiable at a glance.
assert "parameter cx" in out, out
assert out.startswith("badRequest (parameter cx)"), out
ok(f"names the rejected parameter: {out.split(':')[0]!r}")

# ══════════════════════════════════════════════════════════════════════════
print("\n2. The pre-change body still reads the same way")
# What runs 83-85 actually produced — no location field at all.
NO_LOCATION = {
    "error": {
        "code": 400,
        "message": "Request contains an invalid argument.",
        "errors": [{"message": "Request contains an invalid argument.",
                    "domain": "global", "reason": "badRequest"}],
        "status": "INVALID_ARGUMENT",
    }
}
out = search._google_error_reason(FakeResponse(NO_LOCATION))
print(f"      → {out}")
assert out.startswith("badRequest"), out
assert "Request contains an invalid argument." in out
assert "(" not in out.split(":")[0], "no location => no empty parentheses"
assert "global" in out and "INVALID_ARGUMENT" in out
ok("no location => no empty parentheses, and domain/status still surface")

# ══════════════════════════════════════════════════════════════════════════
print("\n3. The 429 quota body is unchanged in substance")
QUOTA_429 = {
    "error": {
        "code": 429,
        "message": ("Quota exceeded for quota metric 'Queries' and limit "
                    "'Queries per day' of service 'customsearch.googleapis.com'."),
        "errors": [{"message": "Quota exceeded.", "domain": "global",
                    "reason": "rateLimitExceeded"}],
        "status": "RESOURCE_EXHAUSTED",
    }
}
out = search._google_error_reason(FakeResponse(QUOTA_429))
print(f"      → {out}")
assert out.startswith("rateLimitExceeded"), out
assert "Queries per day" in out
ok("429 still reads 'rateLimitExceeded: Quota exceeded ... Queries per day'")

# ══════════════════════════════════════════════════════════════════════════
print("\n4. Callers relying on falsiness are unaffected")
for label, resp in (
    ("unparseable body", FakeResponse(None, raises=True)),
    ("empty json",       FakeResponse({})),
    ("null json",        FakeResponse(None)),
    ("no error key",     FakeResponse({"items": []})),
    ("empty error",      FakeResponse({"error": {}})),
    ("error not a dict", FakeResponse({"error": "boom"})),
):
    out = search._google_error_reason(resp)
    assert out == "", f"{label} should yield '' so `or fallback` works, got {out!r}"
ok("6 malformed bodies all return '' — `or '403 forbidden'` fallbacks intact")

# ══════════════════════════════════════════════════════════════════════════
print("\n5. Odd shapes do not raise")
for label, payload in (
    ("errors not a list",   {"error": {"errors": "nope", "message": "m"}}),
    ("errors entry is str", {"error": {"errors": ["nope"], "message": "m"}}),
    ("no errors[]",         {"error": {"message": "m", "status": "S"}}),
    ("no message",          {"error": {"errors": [{"reason": "r"}]}}),
    ("numeric fields",      {"error": {"errors": [{"reason": 400,
                                                   "location": 7}],
                                       "message": 9}}),
):
    out = search._google_error_reason(FakeResponse(payload))
    assert isinstance(out, str), (label, out)
ok("5 odd payloads handled, all return str")

# Multiple errors[] entries: the first entry with the field wins.
MULTI = {"error": {"message": "m", "errors": [
    {"reason": "badRequest"},
    {"reason": "other", "location": "cx", "locationType": "parameter"},
]}}
out = search._google_error_reason(FakeResponse(MULTI))
assert "badRequest" in out and "parameter cx" in out, out
ok("across multiple errors[] entries, the first non-empty value of each wins")

print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
