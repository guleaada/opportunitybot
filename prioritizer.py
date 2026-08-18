"""
prioritizer.py — rank discovered candidates before the MAX_PER_DAY slice.

Discovery returns candidates in provider order, so the fast job APIs
(remotive, arbeitnow) land at the front of the queue and consume every
analysis slot, while the scholarship/fellowship feeds — which is where
matches are actually expected — get whatever is left over.

This module fixes the ORDER, not the size, of that queue. It is pure local
scoring: no model calls, no network, no I/O. Nothing is dropped — a low
scorer stays in the pool, it just ranks below the cut.

Everything tunable lives in the constants below; the logic underneath them
should not need editing to change behaviour.
"""

import re

# ── Source tiers ───────────────────────────────────────────────────────────
# Where a candidate came from is the strongest available signal, because it
# is known before a single byte of the page is read.
TIER_SEED = 3.0               # hand-curated official whitelist URLs
TIER_OPPORTUNITY_FEED = 3.0   # RSS feeds that publish fellowships/scholarships
TIER_SEARCH = 1.5             # Google CSE — mixed quality
TIER_JOB_FEED = 0.5           # RSS feeds tagged as remote_jobs / dev_jobs
TIER_JOB_API = 0.0           # generic job boards (remotive, arbeitnow)
TIER_UNKNOWN = 1.0            # provider we do not recognise

# ── Title signals ──────────────────────────────────────────────────────────
# Per-term weights, each capped so one keyword-stuffed title cannot outrank
# a whole tier on its own.
TITLE_BOOST_PER_TERM = 1.0
TITLE_BOOST_CAP = 3.0
TITLE_PENALTY_PER_TERM = 1.0
TITLE_PENALTY_CAP = 3.0

# Matched as word-prefixes, so "grant" also matches "grants" and "engineer"
# also matches "engineers"/"engineering".
BOOST_TERMS = (
    "fellowship", "scholarship", "grant", "fully funded",
    "call for applications", "residency", "programme", "award", "open call",
)
PENALTY_TERMS = (
    "engineer", "developer", "designer", "sales", "copywriter",
    "assistant", "specialist", "manager", "freelance", "contractor",
)

# How many ranked candidates to print after sorting, as proof the sort ran.
TOP_N_LOGGED = 5


def _prefix_pattern(terms):
    """One regex matching any term at a word boundary (prefix match)."""
    return re.compile(r"\b(?:%s)" % "|".join(re.escape(t) for t in terms))


_BOOST_RE = _prefix_pattern(BOOST_TERMS)
_PENALTY_RE = _prefix_pattern(PENALTY_TERMS)


def _api_source_names():
    """Names of the generic job APIs, read from the registry when available."""
    try:
        import api_sources
        return set(api_sources.REGISTRY)
    except Exception:
        return set()


def source_tier(result, job_feed_domains=(), opportunity_feed_domains=(),
                api_source_names=None) -> float:
    """Tier score for where this candidate came from."""
    source = (getattr(result, "source", "") or "").strip().lower()
    if not source:
        return TIER_UNKNOWN
    if source == "seed":
        return TIER_SEED
    if api_source_names is None:
        api_source_names = _api_source_names()
    if source in api_source_names:
        return TIER_JOB_API
    if source in {d.lower() for d in job_feed_domains}:
        return TIER_JOB_FEED
    if source in {d.lower() for d in opportunity_feed_domains}:
        return TIER_OPPORTUNITY_FEED
    # A domain we did not publish a feed for — a Google CSE result.
    return TIER_SEARCH


def title_adjustment(title: str) -> float:
    """Boost for opportunity vocabulary, penalty for job-posting vocabulary."""
    text = (title or "").lower()
    if not text:
        return 0.0
    boost = min(len(set(_BOOST_RE.findall(text))) * TITLE_BOOST_PER_TERM,
                TITLE_BOOST_CAP)
    penalty = min(len(set(_PENALTY_RE.findall(text))) * TITLE_PENALTY_PER_TERM,
                  TITLE_PENALTY_CAP)
    return boost - penalty


def score_candidate(result, job_feed_domains=(), opportunity_feed_domains=(),
                    api_source_names=None) -> float:
    """Priority score for one candidate. Higher ranks earlier. No side effects."""
    tier = source_tier(result, job_feed_domains, opportunity_feed_domains,
                       api_source_names)
    return round(tier + title_adjustment(getattr(result, "title", "")), 2)


def rank(candidates, job_feed_domains=(), opportunity_feed_domains=(),
         api_source_names=None):
    """[(score, result)] best first. Stable: equal scores keep discovery order.

    Every candidate is returned — this reorders the queue, it never filters.
    """
    if api_source_names is None:
        api_source_names = _api_source_names()
    scored = [(score_candidate(r, job_feed_domains, opportunity_feed_domains,
                               api_source_names), r)
              for r in candidates]
    # sorted() is stable, so candidates that tie keep their discovery order.
    return sorted(scored, key=lambda pair: pair[0], reverse=True)


def prioritize(candidates, job_feed_domains=(), opportunity_feed_domains=(),
               api_source_names=None):
    """The ranked candidate list, same length as the input."""
    return [r for _, r in rank(candidates, job_feed_domains,
                               opportunity_feed_domains, api_source_names)]


def top_lines(scored, n=TOP_N_LOGGED) -> list:
    """Log lines showing the highest-ranked candidates and their scores."""
    if not scored:
        return ["🥇 Priority ranking: no candidates to rank"]
    lines = [f"🥇 Top {min(n, len(scored))} of {len(scored)} candidates "
             f"after priority sort:"]
    for i, (score, r) in enumerate(scored[:n], 1):
        title = (getattr(r, "title", None) or "(untitled)")[:64]
        source = getattr(r, "source", "") or "?"
        lines.append(f"   {i}. {score:+.2f}  [{source}]  {title}")
    return lines
