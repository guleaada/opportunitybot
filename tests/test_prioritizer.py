#!/usr/bin/env python3
"""Candidate prioritisation: rank before the MAX_PER_DAY slice.

Run 83 spent its 15 analysis slots on job listings that were all correctly
rejected, while 76 fellowship-feed posts got what was left. These tests use
that run's real candidate mix.

Run:  python tests/test_prioritizer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prioritizer as P
from search import SearchResult

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  ✅ {msg}")


JOB_FEEDS = {"weworkremotely.com", "remoteok.com", "www.python.org",
             "stackoverflow.com", "jobs.github.com"}
OPP_FEEDS = {"opportunitiescorners.com", "www.opportunitiesforafricans.com",
             "opportunitydesk.org", "www.opportunitiesforyouth.org",
             "opportunitiesforyoungpeople.com"}
APIS = {"remotive", "arbeitnow"}


def r(title, source):
    return SearchResult(title=title, url=f"https://x/{abs(hash(title))}",
                        source=source)


def rank(cands):
    return P.rank(cands, job_feed_domains=JOB_FEEDS,
                  opportunity_feed_domains=OPP_FEEDS, api_source_names=APIS)


# Real titles from production runs 82/83, in the order discovery produced them.
POOL = [
    r("Patient Care Specialist", "remotive"),
    r("Sales Jedi (Account Executive)", "arbeitnow"),
    r("Freelance Copywriter", "remotive"),
    r("Senior Graphic Designer", "arbeitnow"),
    r("Agentic Python Engineer, Evaboot", "www.python.org"),
    r("Senior AI-Augmented Full Stack Developer, SureSwift Capital", "weworkremotely.com"),
    r("INTERPOL Enhanced Secondment Programme 2026: Applications Open",
      "opportunitydesk.org"),
    r("University of British Columbia Four Year Doctoral Fellowship 2026: Fully Funded",
      "www.opportunitiesforafricans.com"),
    r("CRCA Emerging Conflict Analysts Fellowship 2026", "opportunitiescorners.com"),
    r("Chevening Scholarships", "seed"),
    r("Some Random Page About Grants", "randomblog.example"),
]

# ══════════════════════════════════════════════════════════════════════════
print("\n1. Fellowship/scholarship candidates rank above job listings")
ranked = rank(POOL)
titles = [res.title for _, res in ranked]
top5 = titles[:5]
for t in top5:
    assert not any(w in t.lower() for w in
                   ("specialist", "jedi", "copywriter", "designer", "engineer")), \
        f"job listing {t!r} reached the top 5: {top5}"
ok(f"top 5 are all opportunities: {[t[:34] for t in top5]}")

bottom = titles[-4:]
assert "Patient Care Specialist" in bottom, titles
assert "Freelance Copywriter" in bottom, titles
ok(f"generic job-API listings sank to the bottom: {[t[:28] for t in bottom]}")

# ══════════════════════════════════════════════════════════════════════════
print("\n2. Nothing is dropped")
assert len(ranked) == len(POOL), (len(ranked), len(POOL))
assert {id(res) for _, res in ranked} == {id(x) for x in POOL}
ok(f"all {len(POOL)} candidates still in the pool, only reordered")

# ══════════════════════════════════════════════════════════════════════════
print("\n3. Source tiers")
assert P.source_tier(r("x", "seed"), JOB_FEEDS, OPP_FEEDS, APIS) == P.TIER_SEED
assert P.source_tier(r("x", "opportunitydesk.org"), JOB_FEEDS, OPP_FEEDS, APIS) \
    == P.TIER_OPPORTUNITY_FEED
assert P.source_tier(r("x", "randomblog.example"), JOB_FEEDS, OPP_FEEDS, APIS) \
    == P.TIER_SEARCH
assert P.source_tier(r("x", "www.python.org"), JOB_FEEDS, OPP_FEEDS, APIS) \
    == P.TIER_JOB_FEED
assert P.source_tier(r("x", "remotive"), JOB_FEEDS, OPP_FEEDS, APIS) == P.TIER_JOB_API
assert P.source_tier(r("x", "ARBEITNOW"), JOB_FEEDS, OPP_FEEDS, APIS) == P.TIER_JOB_API
assert P.source_tier(r("x", ""), JOB_FEEDS, OPP_FEEDS, APIS) == P.TIER_UNKNOWN
ok("seed=3.0  opportunity feed=3.0  google=1.5  job feed=0.5  job api=0.0")

# ══════════════════════════════════════════════════════════════════════════
print("\n4. Title boosts and penalties")
assert P.title_adjustment("Fully Funded Fellowship") > 0
assert P.title_adjustment("Senior Software Engineer") < 0
assert P.title_adjustment("Call for Applications: Climate Grant") > 0
assert P.title_adjustment("") == 0.0
assert P.title_adjustment(None) == 0.0
# Prefix matching catches plurals and derived forms.
assert P.title_adjustment("Scholarships") > 0
assert P.title_adjustment("Engineering Manager") < 0
# Caps hold: a keyword-stuffed title cannot run away.
stuffed = P.title_adjustment("fellowship scholarship grant award residency open call")
assert stuffed == P.TITLE_BOOST_CAP, stuffed
assert P.title_adjustment("engineer developer designer sales manager") \
    == -P.TITLE_PENALTY_CAP
ok(f"boost capped at +{P.TITLE_BOOST_CAP}, penalty at -{P.TITLE_PENALTY_CAP}")

# A job title on a good feed still loses to an opportunity on the same feed.
same_feed = rank([r("Senior Backend Engineer", "opportunitydesk.org"),
                  r("Fully Funded Fellowship 2026", "opportunitydesk.org")])
assert same_feed[0][1].title.startswith("Fully Funded"), same_feed
ok("within one source, title signal decides the order")

# ══════════════════════════════════════════════════════════════════════════
print("\n5. Stability and robustness")
tie = [r("Untitled A", "opportunitydesk.org"), r("Untitled B", "opportunitydesk.org"),
       r("Untitled C", "opportunitydesk.org")]
assert [x.title for _, x in rank(tie)] == ["Untitled A", "Untitled B", "Untitled C"]
ok("equal scores keep discovery order (stable sort)")

assert rank([]) == []
assert P.top_lines([])[0].startswith("🥇")
odd = rank([SearchResult(title=None, url="https://x", source=None)])
assert len(odd) == 1 and isinstance(odd[0][0], float)
ok("empty pool, None title and None source all handled")

# ══════════════════════════════════════════════════════════════════════════
print("\n6. Log line")
lines = P.top_lines(ranked)
assert len(lines) == 1 + min(P.TOP_N_LOGGED, len(ranked))
assert "after priority sort" in lines[0]
for line in lines[1:]:
    print(f"      {line.strip()}")
ok(f"top_lines() prints {P.TOP_N_LOGGED} ranked candidates with scores")

# ══════════════════════════════════════════════════════════════════════════
print("\n7. No model or network calls")
import model_router
import prioritizer as _p
src = Path(_p.__file__).read_text()
for banned in ("call_model", "requests", "urllib", "import search",
               "socket", "open("):
    assert banned not in src, f"prioritizer references {banned!r}"
ok("prioritizer.py contains no model, network or file-I/O calls")

print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
