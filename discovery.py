"""
discovery.py — provider-based candidate discovery.

Replaces the "fire all 73 Google queries, then RSS" block in run_scan with a
provider layer:

    SearchProvider (Google, budgeted + rotated + 429 breaker)
    RSSProvider    (feed health, permanent-404 disabling)
    APIProvider    (interface; disabled until a source is configured)
    SeedProvider   (whitelist official URLs)
        -> normalize -> deduplicate -> classify -> existing pipeline

Everything downstream is untouched: providers emit the same SearchResult
objects run_scan already consumes.

Persistent state lives in data/ so GitHub Actions runs do not reset it:
    discovery_state.json  — query rotation cursor + per-category performance
    provider_health.json  — per-provider and per-feed health counters
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import taxonomy
from search import SearchResult

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
STATE_PATH = DATA_DIR / "discovery_state.json"
HEALTH_PATH = DATA_DIR / "provider_health.json"

# ── Discovery budget (all configurable; no hard-coded pricing assumptions) ──
GOOGLE_MAX_QUERIES_PER_SCAN = int(os.getenv("GOOGLE_MAX_QUERIES_PER_SCAN", "20"))
GOOGLE_MAX_QUERIES_PER_CATEGORY = int(
    os.getenv("GOOGLE_MAX_QUERIES_PER_CATEGORY", "2"))
SEED_MAX_PER_SCAN = int(os.getenv("SEED_MAX_PER_SCAN", "25"))
RSS_MAX_FEEDS_PER_SCAN = int(os.getenv("RSS_MAX_FEEDS_PER_SCAN", "50"))

# Categories searched EVERY day regardless of rotation — highest value for a
# working AI professional, and the ones with time-sensitive deadlines.
#
# remote_jobs and dev_jobs were half of this list and are now out. They were
# the single largest consumer of the daily query budget while producing the
# generic weworkremotely listings that the eligibility gate then rejected as
# off-profile — the target is funding, not employment. They stay in the
# rotation pool, so nothing is lost, they just no longer crowd out the
# categories this profile can actually win.
ALWAYS_ON_CATEGORIES = [
    c.strip() for c in os.getenv(
        "ALWAYS_ON_CATEGORIES",
        "fellowships,grants,agritech_climate,founder_fellowships").split(",")
    if c.strip()
]

# A provider this unhealthy gets its budget cut for the next scan.
UNHEALTHY_FAILURE_RATIO = float(os.getenv("UNHEALTHY_FAILURE_RATIO", "0.8"))
# Consecutive 404s before a feed is treated as permanently gone.
FEED_DEAD_AFTER_404 = int(os.getenv("FEED_DEAD_AFTER_404", "3"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text() or "null") or default
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️  Could not read {path.name} ({e}) — starting fresh.")
    return default


def _write_json(path: Path, data) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    except OSError as e:
        print(f"⚠️  Could not persist {path.name}: {e}")


# ══════════════════════════════════════════════════════════════════════════
# Provider health
# ══════════════════════════════════════════════════════════════════════════
_HEALTH_FIELDS = ("success_count", "failure_count", "429_count", "403_count",
                  "404_count", "timeout_count", "ssl_error_count")


def load_health() -> dict:
    return _read_json(HEALTH_PATH, {})


def save_health(health: dict) -> None:
    _write_json(HEALTH_PATH, health)


def health_for(health: dict, name: str) -> dict:
    entry = health.setdefault(name, {})
    for f in _HEALTH_FIELDS:
        entry.setdefault(f, 0)
    entry.setdefault("last_success", None)
    entry.setdefault("last_failure", None)
    entry.setdefault("status", "active")
    return entry


def record_success(health: dict, name: str, count: int = 1) -> None:
    e = health_for(health, name)
    e["success_count"] += 1
    e["last_success"] = _now()
    e["last_posts"] = count
    e["status"] = "active"


def record_failure(health: dict, name: str, kind: str = "error",
                   http_status=None) -> None:
    """kind: error | 429 | 403 | 404 | timeout | ssl"""
    e = health_for(health, name)
    e["failure_count"] += 1
    e["last_failure"] = _now()
    e["last_error"] = kind
    if http_status is not None:
        e["last_http_status"] = http_status
    key = {"429": "429_count", "403": "403_count", "404": "404_count",
           "timeout": "timeout_count", "ssl": "ssl_error_count"}.get(kind)
    if key:
        e[key] += 1
    # A feed that 404s repeatedly is gone, not flaky — stop requesting it.
    if kind == "404" and e["404_count"] >= FEED_DEAD_AFTER_404:
        e["status"] = "disabled_permanent"
    elif kind in ("403", "ssl"):
        e["status"] = "temporarily_unavailable"


def is_disabled(health: dict, name: str) -> bool:
    return health_for(health, name).get("status") == "disabled_permanent"


def is_unhealthy(health: dict, name: str) -> bool:
    """Mostly-failing provider → reduce its budget next scan."""
    e = health_for(health, name)
    total = e["success_count"] + e["failure_count"]
    if total < 5:
        return False
    return (e["failure_count"] / total) >= UNHEALTHY_FAILURE_RATIO


# ══════════════════════════════════════════════════════════════════════════
# Query budget, prioritisation and rotation
# ══════════════════════════════════════════════════════════════════════════
def load_state() -> dict:
    state = _read_json(STATE_PATH, {})
    state.setdefault("rotation_cursor", 0)
    state.setdefault("category_performance", {})   # category -> candidates found
    state.setdefault("last_run", None)
    return state


def save_state(state: dict) -> None:
    state["last_run"] = _now()
    _write_json(STATE_PATH, state)


def category_priority(category: str, state: dict) -> tuple:
    """Sort key — lower sorts first. Always-on categories lead, then the
    categories that have historically produced the most candidates."""
    always = 0 if category in ALWAYS_ON_CATEGORIES else 1
    produced = (state.get("category_performance") or {}).get(category, 0)
    return (always, -produced, category)


def select_queries(state: dict, budget: int = None,
                   per_category: int = None) -> list:
    """Pick this scan's queries: always-on categories first, then rotate the
    rest so we don't run the identical set every day.

    Returns ``[(category, query), ...]`` bounded by the budget.
    """
    budget = GOOGLE_MAX_QUERIES_PER_SCAN if budget is None else budget
    per_category = (GOOGLE_MAX_QUERIES_PER_CATEGORY if per_category is None
                    else per_category)
    if budget <= 0:
        return []

    categories = sorted(taxonomy.all_categories(),
                        key=lambda c: category_priority(c, state))
    always = [c for c in categories if c in ALWAYS_ON_CATEGORIES]
    rotating = [c for c in categories if c not in ALWAYS_ON_CATEGORIES]

    # Rotate the non-priority categories by a persisted cursor.
    if rotating:
        cursor = int(state.get("rotation_cursor", 0)) % len(rotating)
        rotating = rotating[cursor:] + rotating[:cursor]

    selected = []
    for category in always + rotating:
        if len(selected) >= budget:
            break
        for q in taxonomy.queries_for(category)[:max(1, per_category)]:
            if len(selected) >= budget:
                break
            selected.append((category, q))
    return selected


def advance_rotation(state: dict, step: int = None) -> dict:
    """Move the cursor so tomorrow starts at different categories."""
    rotating = [c for c in taxonomy.all_categories()
                if c not in ALWAYS_ON_CATEGORIES]
    if not rotating:
        return state
    if step is None:
        step = max(1, GOOGLE_MAX_QUERIES_PER_SCAN //
                   max(1, GOOGLE_MAX_QUERIES_PER_CATEGORY))
    state["rotation_cursor"] = (int(state.get("rotation_cursor", 0)) + step) % len(rotating)
    return state


def record_category_yield(state: dict, category: str, found: int) -> None:
    perf = state.setdefault("category_performance", {})
    perf[category] = perf.get(category, 0) + int(found or 0)


# ══════════════════════════════════════════════════════════════════════════
# Providers — each returns a list[SearchResult]; none may raise
# ══════════════════════════════════════════════════════════════════════════
class Provider:
    name = "provider"
    enabled = True

    def discover(self, ctx) -> list:
        raise NotImplementedError


# Search backends this provider knows how to report on. Order matters only
# for locating the summary section; selection happens in main.py.
SEARCH_SUMMARY_KEYS = ("google", "tavily")

GOOGLE_REQUIRED_ENV = ("GOOGLE_CSE_API_KEY", "GOOGLE_CSE_ID")


class SearchProvider(Provider):
    """A web-search backend: budgeted + rotated + circuit-broken.

    Backend-agnostic. The name, the credentials it needs, the search callable
    and the per-scan state object are all injected, so Google and Tavily
    differ only in construction. Defaults reproduce the Google wiring exactly,
    so existing callers and tests are unaffected.

    ``state`` is any object exposing ``reset_state()``, ``stats()`` and
    ``disabled()`` — ``search`` and ``tavily_search`` both do.
    """
    name = "google"

    def __init__(self, search_fn=None, quality_fn=None, name=None,
                 required_env=None, state=None, label=None):
        self._search = search_fn
        self._quality = quality_fn
        self.name = name or type(self).name
        self._required_env = tuple(required_env or GOOGLE_REQUIRED_ENV)
        self._state = state
        self.label = label or self.name.capitalize()

    def _state_mod(self):
        # Imported lazily so discovery keeps importing without the backend.
        if self._state is None:
            import search as search_mod
            self._state = search_mod
        return self._state

    @property
    def enabled(self) -> bool:
        return all(os.getenv(n) for n in self._required_env)

    def discover(self, ctx) -> list:
        out = []
        summary = ctx["summary"].setdefault(self.name, {})
        if not self.enabled:
            missing = [n for n in self._required_env if not os.getenv(n)]
            msg = f"{' / '.join(missing)} not configured"
            # Explicit, not a silent skip — these are the exact variable names
            # the code reads and the workflow passes from repo secrets.
            print(f"⛔ {self.label} discovery disabled — {msg}")
            summary["skipped"] = msg
            summary["status"] = "DISABLED"
            return out

        state = self._state_mod()
        state.reset_state()
        budget = ctx.get("search_budget", ctx.get("google_budget"))
        queries = select_queries(ctx["state"], budget=budget)
        summary["queries_planned"] = len(queries)

        for category, query in queries:
            if state.disabled():
                summary["stopped_early"] = True
                break
            try:
                results = self._search(query, max_results=ctx["per_query"])
            except Exception as e:
                record_failure(ctx["health"], self.name, "error")
                print(f"⚠️  {self.label} query failed ({category}): {e}")
                continue
            record_category_yield(ctx["state"], category, len(results))
            for r in results:
                r.category = getattr(r, "category", None) or category
                if self._quality:
                    r.source_quality = (getattr(r, "source_quality", None)
                                        or self._quality(r.url))
                out.append(r)

        g = state.stats()
        summary.update({
            "attempted": g["attempted"], "successful": g["successful"],
            "429": g["429"], "403": g["403"], "errors": g["other_errors"],
            "candidates": len(out),
            "disabled_reason": g["disabled_reason"],
        })
        if g["successful"]:
            record_success(ctx["health"], self.name, g["successful"])
        for _ in range(g["429"]):
            record_failure(ctx["health"], self.name, "429", 429)
        for _ in range(g["403"]):
            record_failure(ctx["health"], self.name, "403", 403)
        return out


class RSSProvider(Provider):
    """Feed pull with per-feed health; permanently-404 feeds are skipped."""
    name = "rss"

    def __init__(self, fetch_fn):
        self._fetch = fetch_fn

    def discover(self, ctx) -> list:
        try:
            results, per_feed = self._fetch(ctx)
        except Exception as e:
            print(f"⚠️ RSS discovery failed entirely: {e}")
            record_failure(ctx["health"], self.name, "error")
            return []
        s = ctx["summary"]["rss"]
        s["attempted"] = per_feed.get("attempted", 0)
        s["successful"] = per_feed.get("successful", 0)
        s["failed"] = per_feed.get("failed", 0)
        s["skipped_dead"] = per_feed.get("skipped_dead", 0)
        s["candidates"] = len(results)
        if results:
            record_success(ctx["health"], self.name, len(results))
        return results


class APIProvider(Provider):
    """Interface for key-less JSON job/opportunity APIs.

    Intentionally inert until sources are configured via OPPORTUNITY_API_URLS
    (comma-separated). Present so another provider can be added without
    rewriting discovery.
    """
    name = "api"

    def __init__(self, fetch_json=None):
        self._fetch_json = fetch_json

    @property
    def enabled(self) -> bool:
        try:
            import api_sources
            return bool(api_sources.enabled_sources())
        except Exception:
            return bool(os.getenv("OPPORTUNITY_API_URLS", "").strip())

    def discover(self, ctx) -> list:
        s = ctx["summary"]["api"]
        try:
            import api_sources
            names = api_sources.enabled_sources()
            fetch = self._fetch_json or (lambda n: api_sources.fetch(n))
        except Exception as e:
            s["sources"] = 0
            s["skipped"] = f"api_sources unavailable: {e}"
            return []

        if not names:
            s["sources"] = 0
            s["skipped"] = "no API sources enabled"
            return []

        s["sources"] = len(names)
        out, healthy, failed = [], 0, 0
        for name in names:
            key = f"api:{name}"
            if is_disabled(ctx["health"], key):
                continue
            try:
                items = fetch(name) or []
            except Exception as e:
                failed += 1
                record_failure(ctx["health"], key, "error")
                print(f"⚠️  API source failed {name}: {e}")
                continue
            if not items:
                # Reachable-but-empty is still a failed contribution; health
                # tracking is what eventually retires a dead source.
                failed += 1
                record_failure(ctx["health"], key, "error")
                continue
            healthy += 1
            for it in items:
                title = (it.get("title") or "").strip()
                link = (it.get("url") or it.get("link") or "").strip()
                if not title or not link:
                    continue
                out.append(SearchResult(
                    title=title, url=link,
                    snippet=(it.get("description") or it.get("snippet") or "")[:2000],
                    source=name, category=it.get("category")))
            record_success(ctx["health"], key, len(items))
        s["healthy"] = healthy
        s["failed"] = failed
        s["candidates"] = len(out)
        return out


class SeedProvider(Provider):
    """Official whitelist URLs — cheap, always available, bounded."""
    name = "seed"

    def __init__(self, seeds_fn):
        self._seeds = seeds_fn

    def discover(self, ctx) -> list:
        s = ctx["summary"]["seed"]
        try:
            seeds = self._seeds() or []
        except Exception as e:
            print(f"⚠️  Seed provider failed: {e}")
            record_failure(ctx["health"], self.name, "error")
            s["attempted"] = 0
            return []
        seeds = seeds[:SEED_MAX_PER_SCAN]
        s["attempted"] = len(seeds)
        out = [SearchResult(title=sd.get("name", ""), url=sd.get("url", ""),
                            source="seed")
               for sd in seeds if sd.get("url")]
        s["candidates"] = len(out)
        if out:
            record_success(ctx["health"], self.name, len(out))
        return out


# ══════════════════════════════════════════════════════════════════════════
# Normalize / deduplicate
# ══════════════════════════════════════════════════════════════════════════
def _canonical(url: str) -> str:
    try:
        from notification import canonical_url
        return canonical_url(url)
    except Exception:
        return (url or "").strip().lower().rstrip("/")


def normalize_and_dedupe(batches) -> list:
    """Flatten provider batches, drop entries without a URL, and keep the
    first occurrence of each canonical URL. The richest snippet wins when the
    same URL arrives from several providers."""
    out, index = [], {}
    for results in batches:
        for r in results or []:
            url = getattr(r, "url", None)
            if not url:
                continue
            key = _canonical(url)
            if not key:
                continue
            if key in index:
                kept = out[index[key]]
                # Prefer whichever carries more analysable text.
                if len(getattr(r, "snippet", "") or "") > len(getattr(kept, "snippet", "") or ""):
                    kept.snippet = r.snippet
                kept.category = getattr(kept, "category", None) or getattr(r, "category", None)
                continue
            index[key] = len(out)
            out.append(r)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════
def new_summary(search_name: str = "google") -> dict:
    return {
        search_name: {"attempted": 0, "successful": 0, "429": 0, "403": 0,
                      "errors": 0, "candidates": 0, "queries_planned": 0},
        "rss": {"attempted": 0, "successful": 0, "failed": 0,
                "skipped_dead": 0, "candidates": 0},
        "api": {"sources": 0, "healthy": 0, "failed": 0, "candidates": 0},
        "seed": {"attempted": 0, "candidates": 0},
        "total": {"raw": 0, "after_dedupe": 0},
    }


def search_budget_for(health: dict, name: str = "google") -> int:
    """Cut the search budget when that backend has been failing — lean on
    RSS/API instead."""
    budget = GOOGLE_MAX_QUERIES_PER_SCAN
    if is_unhealthy(health, name):
        budget = max(1, budget // 4)
        print(f"⚠️  {name.capitalize()} looks unhealthy — "
              f"reducing query budget to {budget}")
    return budget


# Retained name for existing callers/tests.
google_budget_for = search_budget_for


def search_provider_name(providers) -> str:
    """Which search backend is in this run's provider list."""
    for p in providers or []:
        if isinstance(p, SearchProvider):
            return p.name
    return "google"


def run_discovery(providers, per_query: int = 8) -> tuple:
    """Run every provider, normalize, dedupe. Returns (candidates, summary).

    A provider that fails is isolated: the others still contribute.
    """
    state = load_state()
    health = load_health()
    search_name = search_provider_name(providers)
    summary = new_summary(search_name)
    budget = search_budget_for(health, search_name)
    ctx = {"state": state, "health": health, "summary": summary,
           "per_query": per_query, "search_budget": budget,
           # Retained key so anything still reading it keeps working.
           "google_budget": budget}

    batches = []
    for p in providers:
        try:
            batches.append(p.discover(ctx))
        except Exception as e:
            print(f"⚠️  Provider {getattr(p, 'name', '?')} failed: {e}")
            record_failure(health, getattr(p, "name", "unknown"), "error")
            batches.append([])

    summary["total"]["raw"] = sum(len(b or []) for b in batches)
    candidates = normalize_and_dedupe(batches)
    summary["total"]["after_dedupe"] = len(candidates)

    advance_rotation(state)
    save_state(state)
    save_health(health)
    return candidates, summary


# Coverage thresholds — diagnostic only. Candidates are NEVER manufactured to
# hit these; they simply describe how thin the real yield was.
COVERAGE_WARN_BELOW = int(os.getenv("DISCOVERY_WARN_BELOW", "20"))
COVERAGE_CRITICAL_BELOW = int(os.getenv("DISCOVERY_CRITICAL_BELOW", "5"))


def search_section(summary: dict):
    """(name, section) for whichever search backend this summary describes."""
    for key in SEARCH_SUMMARY_KEYS:
        if isinstance(summary.get(key), dict):
            return key, summary[key]
    return "google", {}


def search_status(g: dict) -> str:
    if g.get("skipped"):
        return "DISABLED"
    if g.get("429", 0) and not g.get("successful", 0):
        return "RATE_LIMITED"
    if g.get("attempted", 0) and not g.get("successful", 0):
        return "FAILED"
    if g.get("successful", 0):
        return "RATE_LIMITED" if g.get("429", 0) else "HEALTHY"
    return "IDLE"


# Retained name for existing callers/tests.
google_status = search_status


def coverage_level(raw: int) -> str:
    if raw < COVERAGE_CRITICAL_BELOW:
        return "CRITICAL"
    if raw < COVERAGE_WARN_BELOW:
        return "WARNING"
    return "OK"


def format_summary(summary: dict, new_candidates=None) -> list:
    """Report lines that make it obvious whether discovery or downstream
    filtering is the bottleneck, and when coverage is too thin to trust."""
    search_name, g = search_section(summary)
    r = summary["rss"]
    a, s, t = summary["api"], summary["seed"], summary["total"]
    raw = t.get("raw", 0)

    # Padded so "Google:" and "Tavily:" align identically to before.
    label = f"{search_name.capitalize()}:"
    lines = [
        "🔭 DISCOVERY HEALTH",
        f"   {label:8} status {search_status(g)}  •  "
        f"planned {g.get('queries_planned', 0)}, attempted {g.get('attempted', 0)}, "
        f"ok {g.get('successful', 0)}, 429 {g.get('429', 0)}, "
        f"403 {g.get('403', 0)} → {g.get('candidates', 0)} candidates",
    ]
    if g.get("skipped"):
        lines.append(f"            reason: {g['skipped']}")
    if g.get("disabled_reason"):
        lines.append(f"            stopped: {g['disabled_reason']}")

    attempted_feeds = r.get("attempted", 0)
    lines += [
        f"   RSS:     active feeds {r.get('successful', 0)}/{attempted_feeds}"
        f"  •  failed {r.get('failed', 0)}, dead-skipped "
        f"{r.get('skipped_dead', 0)} → {r.get('candidates', 0)} posts",
        f"   APIs:    healthy {a.get('healthy', 0)}, failed {a.get('failed', 0)}"
        f" of {a.get('sources', 0)} → {a.get('candidates', 0)} candidates",
    ]
    if a.get("skipped"):
        lines.append(f"            reason: {a['skipped']}")
    lines += [
        f"   Seeds:   active {s.get('candidates', 0)}/{s.get('attempted', 0)}"
        f" → {s.get('candidates', 0)} candidates",
        f"   TOTAL:   raw {raw} → deduplicated {t.get('after_dedupe', 0)}"
        + (f" → new {new_candidates}" if new_candidates is not None else ""),
    ]

    level = coverage_level(raw)
    if level == "CRITICAL":
        lines.append(f"   🚨 CRITICAL: Discovery providers are mostly "
                     f"unavailable (raw {raw} < {COVERAGE_CRITICAL_BELOW}).")
    elif level == "WARNING":
        lines.append(f"   ⚠️  WARNING: Discovery coverage is currently low "
                     f"(raw {raw} < {COVERAGE_WARN_BELOW}).")
    return lines
