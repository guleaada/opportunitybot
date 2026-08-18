#!/usr/bin/env python3
"""SearchProvider is backend-agnostic; Tavily is preferred when configured.

Google's Custom Search JSON API is unusable on this Cloud project (403
PERMISSION_DENIED), so Tavily replaces it. Google's implementation is kept
intact and stays the fallback — these tests pin both paths, and pin that
nothing about RSS/API/Seed discovery moved.

Run:  python tests/test_search_provider_wiring.py
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import discovery
import rate_limiter
import search
import tavily_search as tv
from search import SearchResult

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  ✅ {msg}")


TAVILY_ENV = {"TAVILY_API_KEY": "tvly-dev-UNITTEST"}
GOOGLE_ENV = {"GOOGLE_CSE_API_KEY": "k", "GOOGLE_CSE_ID": "c"}


def select(env):
    """(provider, enabled) from main._search_provider() under a given env.

    `enabled` is read INSIDE the patched environment — it is a property that
    consults os.getenv, so evaluating it afterwards would read the real env.
    """
    import main
    with patch.dict(os.environ, env, clear=True):
        prov = main._search_provider()
        return prov, prov.enabled


# ══════════════════════════════════════════════════════════════════════════
print("\n1. Provider selection")
p, enabled = select(TAVILY_ENV)
assert p.name == "tavily" and enabled is True
assert p._required_env == ("TAVILY_API_KEY",), p._required_env
assert p._state is tv, "Tavily provider must carry the Tavily state module"
assert p._search is tv.tavily_search, "must reuse the Step 1 implementation"
ok("TAVILY_API_KEY set → tavily, wired to tavily_search.tavily_search")

p, _ = select({**TAVILY_ENV, **GOOGLE_ENV})
assert p.name == "tavily", "Tavily must win when both are configured"
ok("both configured → tavily preferred over google")

p, enabled = select(GOOGLE_ENV)
assert p.name == "google" and enabled is True
assert p._required_env == ("GOOGLE_CSE_API_KEY", "GOOGLE_CSE_ID")
ok("Tavily absent, Google creds present → google")

for env, label in (({}, "neither"), ({"GOOGLE_CSE_API_KEY": "k"}, "partial google")):
    p, enabled = select(env)
    assert p.name == "google" and enabled is False, (label, p.name, enabled)
ok("neither / partial credentials → search provider reports DISABLED")

# ══════════════════════════════════════════════════════════════════════════
print("\n2. The default construction is still the Google one")
p = discovery.SearchProvider(search_fn=lambda q, max_results=8: [])
assert p.name == "google"
assert p._required_env == ("GOOGLE_CSE_API_KEY", "GOOGLE_CSE_ID")
with patch.dict(os.environ, GOOGLE_ENV, clear=True):
    assert p.enabled is True
assert p._state_mod() is search, "default state module must be search.py"
ok("SearchProvider(search_fn=…) alone still means Google — unchanged contract")

# Google's own module satisfies the state interface via aliases.
for fn in ("reset_state", "stats", "disabled"):
    assert callable(getattr(search, fn)), fn
assert search.reset_state is search.reset_google_state
assert search.stats is search.google_stats
assert search.disabled is search.google_disabled
ok("search.py exposes reset_state/stats/disabled as aliases, originals intact")

# ══════════════════════════════════════════════════════════════════════════
print("\n3. Tavily state drives reset / disabled / stats")
calls = []


class FakeState:
    def reset_state(self):
        calls.append("reset")
    def disabled(self):
        calls.append("disabled")
        return False
    def stats(self):
        calls.append("stats")
        return {"attempted": 2, "successful": 2, "429": 0, "403": 0,
                "other_errors": 0, "disabled_reason": ""}


def run_discover(name, env, required, state, results):
    prov = discovery.SearchProvider(
        search_fn=lambda q, max_results=8: list(results),
        name=name, required_env=required, state=state)
    ctx = {"state": {}, "health": {}, "summary": discovery.new_summary(name),
           "per_query": 3, "search_budget": 2}
    with patch.dict(os.environ, env, clear=True), \
         patch.object(discovery, "select_queries",
                      return_value=[("fellowships", "q1"), ("grants", "q2")]), \
         patch.object(discovery, "record_category_yield"):
        out = prov.discover(ctx)
    return out, ctx


calls.clear()
out, ctx = run_discover("tavily", TAVILY_ENV, ("TAVILY_API_KEY",), FakeState(), [])
assert calls[0] == "reset", calls
assert "disabled" in calls and "stats" in calls, calls
ok(f"discover() calls the injected state: {calls}")

# ══════════════════════════════════════════════════════════════════════════
print("\n4. Summary and health are keyed by the selected backend")
hits = [SearchResult(title="Fellowship", url="https://example.org/f",
                     snippet="s", source="example.org")]
out, ctx = run_discover("tavily", TAVILY_ENV, ("TAVILY_API_KEY",),
                        FakeState(), hits)
assert "tavily" in ctx["summary"], list(ctx["summary"])
assert "google" not in ctx["summary"], "google section must not appear"
assert ctx["summary"]["tavily"]["successful"] == 2
assert ctx["summary"]["tavily"]["candidates"] == len(out) == 2
assert "tavily" in ctx["health"] and "google" not in ctx["health"], ctx["health"]
ok("summary and provider_health both recorded under 'tavily'")

name, section = discovery.search_section(ctx["summary"])
assert name == "tavily", name
line = discovery.format_summary(ctx["summary"])[1]
assert line.startswith("   Tavily:  status"), repr(line)
ok(f"report line reads {line.strip()[:34]!r}")

# Google keeps its own line, byte-identical alignment.
gline = discovery.format_summary(discovery.new_summary("google"))[1]
assert gline.startswith("   Google:  status"), repr(gline)
ok("google summary still renders '   Google:  status …' unchanged")

# ══════════════════════════════════════════════════════════════════════════
print("\n5. Results flow through discovery with category and quality applied")
prov = discovery.SearchProvider(
    search_fn=lambda q, max_results=8: [
        SearchResult(title="T", url="https://good.example/x", snippet="s",
                     source="good.example")],
    quality_fn=lambda url: "tier1",
    name="tavily", required_env=("TAVILY_API_KEY",), state=FakeState())
ctx = {"state": {}, "health": {}, "summary": discovery.new_summary("tavily"),
       "per_query": 3, "search_budget": 1}
with patch.dict(os.environ, TAVILY_ENV, clear=True), \
     patch.object(discovery, "select_queries",
                  return_value=[("fellowships", "q1")]), \
     patch.object(discovery, "record_category_yield"):
    out = prov.discover(ctx)
assert len(out) == 1
assert out[0].category == "fellowships", out[0].category
assert out[0].source_quality == "tier1", out[0].source_quality
ok("category and source_quality applied exactly as for Google")

# ══════════════════════════════════════════════════════════════════════════
print("\n6. Rate limiter registration")
assert "tavily" in rate_limiter.LIMITS, rate_limiter.known_providers()
lim = rate_limiter.LIMITS["tavily"]
assert lim["rpm"] == tv.TAVILY_REQUESTS_PER_MINUTE
assert lim["rpd"] == tv.TAVILY_REQUESTS_PER_DAY
assert "tavily" in rate_limiter.known_providers()
# Registration must create the quota deques too — the parallel-dict bug.
assert rate_limiter.seconds_until_available("tavily") == 0.0
assert "tavily" in rate_limiter.snapshot()
ok(f"registered via register_provider: rpm={lim['rpm']} rpd={lim['rpd']}, "
   f"deques live")

# Google's own limits are untouched.
assert rate_limiter.LIMITS["google"]["rpm"] == 10
ok("google rate limits unchanged")

# ══════════════════════════════════════════════════════════════════════════
print("\n7. Nothing else in discovery moved")
for cls in ("RSSProvider", "APIProvider", "SeedProvider"):
    assert hasattr(discovery, cls), cls
assert discovery.google_budget_for is discovery.search_budget_for
assert discovery.google_status is discovery.search_status
ok("RSS/API/Seed providers intact; google_budget_for and google_status "
   "still resolve")

# A run with no search provider at all still produces a usable summary.
s = discovery.new_summary("google")
assert set(s) >= {"google", "rss", "api", "seed", "total"}, list(s)
lines = discovery.format_summary(s)
assert any("RSS:" in ln for ln in lines) and any("APIs:" in ln for ln in lines)
ok("summary still carries rss/api/seed/total sections and renders them")

print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
