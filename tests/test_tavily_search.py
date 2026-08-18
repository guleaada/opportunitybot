#!/usr/bin/env python3
"""tavily_search: request shape, mapping, error handling, per-scan state.

Google CSE is unusable for this project (403 PERMISSION_DENIED — the Cloud
project has no access to the Custom Search JSON API), so Tavily is the
replacement backend. This module is a drop-in for search.web_search and must
honour the same contracts: never raise, return [] on failure, and keep a
per-scan breaker.

Every HTTP call is mocked. No real Tavily request is ever made.

Run:  python tests/test_tavily_search.py
"""
import os
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tavily_search as tv
from search import SearchResult

PASSED = []
FAKE_KEY = "tvly-dev-UNITTESTKEY0000000000000000"


def ok(msg):
    PASSED.append(msg)
    print(f"  ✅ {msg}")


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None, bad_json=False):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self._bad_json = bad_json

    def json(self):
        if self._bad_json:
            raise ValueError("Expecting value: line 1 column 1 (char 0)")
        return self._payload


# A real-shaped Tavily 200 body (field names per tavily-python / langchain-tavily).
def body(n=3):
    return {
        "query": "fully funded fellowship",
        "follow_up_questions": None,
        "answer": None,
        "images": [],
        "results": [
            {"title": f"Result {i}",
             "url": f"https://www.OpportunityDesk{i}.org/page?a=1",
             "content": f"snippet body {i}",
             "score": 0.9 - i / 100,
             "raw_content": None}
            for i in range(n)
        ],
        "response_time": 1.31,
    }


def run(query="fully funded fellowship", max_results=10, responses=None,
        key=FAKE_KEY, reset=True):
    """Call tavily_search with requests.post mocked. Returns (results, mock)."""
    if reset:
        tv.reset_state()
    seq = list(responses or [FakeResponse(200, body())])
    env = dict(os.environ)
    env.pop("TAVILY_API_KEY", None)
    if key is not None:
        env["TAVILY_API_KEY"] = key
    with patch.dict(os.environ, env, clear=True), \
         patch.object(tv, "time") as fake_time, \
         patch.object(tv, "wait_for_quota"), \
         patch.object(tv.requests, "post") as post:
        fake_time.sleep = lambda s: None          # never really sleep
        post.side_effect = seq if len(seq) > 1 else None
        if len(seq) == 1:
            post.return_value = seq[0]
        out = tv.tavily_search(query, max_results=max_results)
    return out, post


# ══════════════════════════════════════════════════════════════════════════
print("\n1. Endpoint, method and authentication placement")
out, post = run()
assert post.call_count == 1, post.call_count
args, kwargs = post.call_args
assert args[0] == "https://api.tavily.com/search", args[0]
assert tv.TAVILY_URL == "https://api.tavily.com/search"
headers = kwargs["headers"]
assert headers["Authorization"] == f"Bearer {FAKE_KEY}", "auth must be a Bearer header"
assert headers["Content-Type"] == "application/json"
ok("POST https://api.tavily.com/search with Authorization: Bearer <key>")

# The key must go in the header, never in the JSON body.
sent = kwargs["json"]
assert "api_key" not in sent and FAKE_KEY not in str(sent), sent
ok("the key is never placed in the request body")

# ══════════════════════════════════════════════════════════════════════════
print("\n2. Query and max_results are sent correctly")
assert sent["query"] == "fully funded fellowship", sent
assert sent["max_results"] == 10, sent
out, post = run(query="grants 2026", max_results=3)
assert post.call_args.kwargs["json"]["query"] == "grants 2026"
assert post.call_args.kwargs["json"]["max_results"] == 3
ok("query and max_results are passed through verbatim")

# Clamped to the documented per-call ceiling, and never below 1.
out, post = run(max_results=999)
assert post.call_args.kwargs["json"]["max_results"] == tv.TAVILY_MAX_RESULTS_PER_CALL
out, post = run(max_results=0)
assert post.call_args.kwargs["json"]["max_results"] == 1
ok(f"clamped to [1, {tv.TAVILY_MAX_RESULTS_PER_CALL}] — no offset/pagination exists")

# ══════════════════════════════════════════════════════════════════════════
print("\n3. max_results is honoured even if Tavily over-returns")
out, _ = run(max_results=2, responses=[FakeResponse(200, body(n=6))])
assert len(out) == 2, len(out)
ok("6 results returned, 2 requested → 2 kept")

# ══════════════════════════════════════════════════════════════════════════
print("\n4. SearchResult mapping")
out, _ = run(max_results=3)
assert all(isinstance(r, SearchResult) for r in out)
assert type(out[0]).__module__ == "search", "must reuse the project's model"
assert out[0].title == "Result 0"
assert out[0].url == "https://www.OpportunityDesk0.org/page?a=1"
assert out[0].snippet == "snippet body 0", out[0].snippet
ok("title, url and content→snippet mapped onto search.SearchResult")

# ══════════════════════════════════════════════════════════════════════════
print("\n5. source is derived from the URL netloc")
assert out[0].source == "www.opportunitydesk0.org", out[0].source
assert tv._source_of("https://Example.COM/a/b") == "example.com"
assert tv._source_of("https://sub.example.co.uk:8443/x") == "sub.example.co.uk:8443"
assert tv._source_of("not a url") == ""
assert tv._source_of("") == ""
# "www." is preserved: the prioritizer's feed-domain sets contain the www form.
assert tv._source_of("https://www.python.org/jobs/") == "www.python.org"
ok("netloc lowercased, 'www.' preserved, junk input yields ''")

# A result with no URL is unusable and must be dropped, not crash.
malformed = {"results": [{"title": "no url", "content": "x"},
                         {"url": "https://ok.example/1", "title": None,
                          "content": None},
                         "not-a-dict", None]}
out, _ = run(responses=[FakeResponse(200, malformed)])
assert len(out) == 1 and out[0].url == "https://ok.example/1", out
assert out[0].title == "" and out[0].snippet == ""
ok("URL-less / non-dict / None entries dropped; None title+content → ''")

# ══════════════════════════════════════════════════════════════════════════
print("\n6. The API key never appears in output")
buf = StringIO()
with patch("sys.stdout", buf):
    run(responses=[FakeResponse(500, {"error": {"message": f"boom {FAKE_KEY}"}})])
    run(responses=[FakeResponse(401, {"error": {"message": "bad key"}})])
    run(responses=[FakeResponse(200, None, bad_json=True)])
printed = buf.getvalue()
assert FAKE_KEY not in printed, printed
assert "<redacted>" in printed, "an echoed key should be scrubbed, not passed through"
ok("even a server echoing the key back is scrubbed before printing")

# The scrubber itself, directly.
with patch.dict(os.environ, {"TAVILY_API_KEY": FAKE_KEY}):
    assert FAKE_KEY not in tv._scrub(f"error with {FAKE_KEY} inside")
ok("_scrub() removes the key from any message")

# ══════════════════════════════════════════════════════════════════════════
print("\n7. Missing API key returns [] without calling out")
for missing in (None, "", "   "):
    tv.reset_state()
    with patch.dict(os.environ, ({} if missing is None
                                 else {"TAVILY_API_KEY": missing}), clear=True), \
         patch.object(tv.requests, "post") as post:
        assert tv.tavily_search("q") == []
        assert post.call_count == 0, "must not fire a request without a key"
ok("unset / empty / whitespace-only key → [], zero requests, no raise")

# ══════════════════════════════════════════════════════════════════════════
print("\n8. Empty and malformed 200 bodies are safe")
for label, payload, bad in (
    ("no results key", {"query": "q"}, False),
    ("results empty",  {"results": []}, False),
    ("results null",   {"results": None}, False),
    ("body is null",   None, False),
    ("body is a list", [1, 2, 3], False),
    ("unparseable",    None, True),
):
    tv.reset_state()
    out, _ = run(responses=[FakeResponse(200, payload, bad_json=bad)], reset=False)
    assert out == [], (label, out)
ok("6 empty/malformed 200 bodies all return [] without raising")

# ══════════════════════════════════════════════════════════════════════════
print("\n9. 403 / 432 / 433 are recorded and disable the provider")
for status in (403, 432, 433):
    tv.reset_state()
    out, _ = run(responses=[FakeResponse(status, {"error": {"message": "plan limit"}})],
                 reset=False)
    s = tv.stats()
    assert out == [], out
    assert s["403"] == 1, (status, s)
    assert s["disabled"] is True and tv.disabled() is True, (status, s)
    assert "plan limit" in s["disabled_reason"], s
    # Once disabled, a further call short-circuits without any request.
    _, post2 = run(reset=False)
    assert post2.call_count == 0, f"{status} did not stop later queries"
ok("403, 432 and 433 each count as 403, disable the scan, and stop further calls")

print("\n10. 401 disables too, naming the variable and not the value")
tv.reset_state()
buf = StringIO()
with patch("sys.stdout", buf):
    out, _ = run(responses=[FakeResponse(401, {"error": {"message": "bad key"}})],
                 reset=False)
s = tv.stats()
assert out == [] and s["disabled"] is True and s["other_errors"] == 1, s
assert "TAVILY_API_KEY" in buf.getvalue() and FAKE_KEY not in buf.getvalue()
ok("401 → disabled, message names TAVILY_API_KEY, never its value")

# ══════════════════════════════════════════════════════════════════════════
print("\n11. 429 is recorded, backs off, and trips the breaker")
tv.reset_state()
out, post = run(responses=[FakeResponse(429, {}, headers={"Retry-After": "1"}),
                           FakeResponse(429, {}),
                           FakeResponse(200, body(n=2))], reset=False)
s = tv.stats()
assert len(out) == 2, out
assert s["429"] == 2, s
assert s["consecutive_429"] == 0, "a success must clear the streak"
assert s["disabled"] is False, s
assert post.call_count == 3, post.call_count
ok("two 429s then a 200 → retried, counted, streak cleared, not disabled")

tv.reset_state()
out, post = run(responses=[FakeResponse(429, {})] * 3, reset=False)
s = tv.stats()
assert out == [] and s["429"] == 3, s
assert s["disabled"] is True, "breaker must trip at TAVILY_MAX_CONSECUTIVE_429"
assert post.call_count == tv.TAVILY_MAX_CONSECUTIVE_429, post.call_count
ok(f"{tv.TAVILY_MAX_CONSECUTIVE_429} consecutive 429s → disabled, no further requests")

# Retry-After is honoured and bounded.
assert tv._retry_after_seconds(FakeResponse(429, {}, {"Retry-After": "5"}), 0) == 5.0
assert tv._retry_after_seconds(FakeResponse(429, {}, {"Retry-After": "9999"}), 0) \
    == tv.TAVILY_BACKOFF_MAX_SECONDS
assert tv._retry_after_seconds(FakeResponse(429, {}, {"Retry-After": "junk"}), 0) \
    == tv.TAVILY_BACKOFF_BASE_SECONDS
assert tv._retry_after_seconds(FakeResponse(429, {}), 2) \
    == min(tv.TAVILY_BACKOFF_BASE_SECONDS * 4, tv.TAVILY_BACKOFF_MAX_SECONDS)
ok("Retry-After honoured, capped, and falls back to exponential backoff")

# ══════════════════════════════════════════════════════════════════════════
print("\n12. Other non-200 responses return [] without disabling")
for status in (400, 404, 500, 502, 503):
    tv.reset_state()
    out, _ = run(responses=[FakeResponse(status, {"error": {"message": "nope"}})],
                 reset=False)
    s = tv.stats()
    assert out == [], (status, out)
    assert s["other_errors"] == 1, (status, s)
    assert s["disabled"] is False, f"{status} must stay retryable next query"
ok("400/404/500/502/503 → [], counted as other_errors, provider stays enabled")

# A transport failure is a failure, not a crash.
tv.reset_state()
with patch.dict(os.environ, {"TAVILY_API_KEY": FAKE_KEY}), \
     patch.object(tv, "wait_for_quota"), \
     patch.object(tv.requests, "post",
                  side_effect=tv.requests.RequestException("conn reset")):
    assert tv.tavily_search("q") == []
assert tv.stats()["other_errors"] == 1
ok("a requests transport error returns [] and is counted, never raised")

# ══════════════════════════════════════════════════════════════════════════
print("\n13. Module state, reset_state(), stats() and disabled()")
tv.reset_state()
blank = tv.stats()
for field in ("attempted", "successful", "429", "403", "other_errors",
              "consecutive_429", "disabled"):
    assert field in blank, f"missing state field {field}"
assert all(blank[f] == 0 for f in ("attempted", "successful", "429", "403",
                                   "other_errors", "consecutive_429"))
assert blank["disabled"] is False and tv.disabled() is False
ok("all seven required state fields present and zeroed after reset_state()")

run(responses=[FakeResponse(200, body(n=2))], reset=False)
s = tv.stats()
assert s["attempted"] == 1 and s["successful"] == 1 and s["results"] == 2, s
s["attempted"] = 999                       # stats() must hand back a copy
assert tv.stats()["attempted"] == 1, "stats() leaked the live dict"
ok("counters advance on success, and stats() returns a copy not the live dict")

tv.reset_state()
assert tv.stats()["attempted"] == 0 and tv.disabled() is False
ok("reset_state() clears counters and re-arms the breaker between scans")

print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
