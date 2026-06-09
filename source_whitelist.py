"""
source_whitelist.py — trusted sources with quality scores (1-10).

Quality tiers:
  10  Official scholarship bodies / government / embassy sites.
   8  Highly reliable program operators & well-known foundations.
   7  Reputable aggregators that link to official pages.
   5  General aggregators (use, but verify against the official source).

Each source carries a ``search_query`` the scanner feeds to web search. The
queries are tuned to the user's profile (Ethiopian national, working pro,
fully-funded masters / fellowships / AI-tech residencies).
"""

from typing import List


class Source:
    def __init__(self, name: str, domain: str, quality: int, search_query: str,
                 category: str = "general"):
        self.name = name
        self.domain = domain
        self.quality = quality
        self.search_query = search_query
        self.category = category

    def __repr__(self):
        return f"<Source {self.name} q={self.quality}>"


# Profile-tuned query fragment reused across sources.
_Q = "fully funded scholarship OR fellowship 2026 2027 Ethiopian developing country international applicants"

SOURCES: List[Source] = [
    # ── Tier 10: official government / body sites ──────────────────────────
    Source("DAAD", "daad.de", 10,
           "site:daad.de scholarship masters fully funded developing countries EPOS 2026"),
    Source("Chevening", "chevening.org", 10,
           "site:chevening.org scholarship eligibility apply 2026 2027"),
    Source("Fulbright Foreign Student", "foreign.fulbrightonline.org", 10,
           "Fulbright foreign student program Ethiopia masters 2026 fully funded"),
    Source("MEXT Japan", "studyinjapan.go.jp", 10,
           "MEXT scholarship Japanese government 2026 research masters international students"),
    Source("Erasmus Mundus", "erasmus-plus.ec.europa.eu", 10,
           "Erasmus Mundus joint masters scholarship 2026 2027 fully funded apply"),
    Source("Australia Awards", "australiaawards.gov.au", 10,
           "Australia Awards scholarship Africa Ethiopia 2026 fully funded"),
    Source("Commonwealth Scholarships", "cscuk.fcdo.gov.uk", 10,
           "Commonwealth scholarship masters developing commonwealth 2026 fully funded"),
    Source("Vanier Canada", "vanier.gc.ca", 10,
           "Vanier Canada graduate scholarship international 2026"),
    Source("Campus France", "campusfrance.org", 10,
           "Campus France Eiffel scholarship masters 2026 international students"),
    Source("Nuffic / StuNed", "nuffic.nl", 10,
           "Netherlands Orange Knowledge OR Holland scholarship masters 2026 international"),
    Source("Swedish Institute", "si.se", 10,
           "Swedish Institute scholarship global professionals 2026 masters fully funded"),
    Source("KGSP Korea", "studyinkorea.go.kr", 10,
           "Global Korea Scholarship GKS graduate 2026 international students"),

    # ── Tier 8: foundations & well-known programs ──────────────────────────
    Source("Obama Foundation Scholars", "obama.org", 8,
           "Obama Foundation Scholars program 2026 leaders apply fully funded"),
    Source("Acumen Fellowship", "acumen.org", 8,
           "Acumen fellowship 2026 social entrepreneur Africa apply"),
    Source("Ashoka Fellowship", "ashoka.org", 8,
           "Ashoka fellowship social entrepreneur 2026 apply"),
    Source("Gates Cambridge", "gatescambridge.org", 8,
           "Gates Cambridge scholarship 2026 international fully funded"),
    Source("Mastercard Foundation Scholars", "mastercardfdn.org", 8,
           "Mastercard Foundation Scholars Program 2026 Africa masters fully funded"),
    Source("Mandela Rhodes / Rhodes", "rhodeshouse.ox.ac.uk", 8,
           "Rhodes scholarship 2026 Africa international fully funded apply"),
    Source("Schwarzman Scholars", "schwarzmanscholars.org", 8,
           "Schwarzman Scholars 2026 masters Tsinghua fully funded apply"),

    # ── Tier 8: high-signal aggregators ────────────────────────────────────
    Source("OpportunityDesk", "opportunitydesk.org", 8,
           "fellowship scholarship residency 2026 fully funded apply international"),

    # ── Tier 7: reputable aggregators ──────────────────────────────────────
    Source("OpportunitiesForAfricans", "opportunitiesforafricans.com", 7,
           "fully funded scholarships fellowships 2026 Africa Ethiopia international"),
    Source("OpportunitiesForYouth", "opportunitiesforyouth.org", 7,
           "fully funded scholarship fellowship 2026 youth international apply"),
    Source("OpportunitiesCorners", "opportunitiescorners.com", 7,
           "fully funded scholarship fellowship 2026 2027 international students apply"),
    Source("All Africa Foundation", "", 7,
           "All Africa Foundation scholarship fellowship 2026 Africa fully funded"),
    Source("ScholarshipRegion", "scholarshipregion.com", 7,
           "fully funded scholarship fellowship 2026 2027 developing countries apply"),
    Source("After School Africa", "afterschoolafrica.com", 7,
           "fully funded scholarship fellowship 2026 Africa international apply"),

    # ── Tier 6: broad aggregators (verify against official source) ──────────
    Source("ScholarshipsCorner", "scholarshipscorner.website", 6,
           f"fully funded scholarships fellowships 2026 2027 Ethiopia Africa {_Q}"),

    # ── Tier 7: AI / tech innovator residencies & accelerators ─────────────
    Source("Tech Residencies (AI)", "", 7,
           "AI technology innovator fellowship residency 2026 fully funded sponsored visa international developing country"),
    Source("Research Fellowships (AI/ML)", "", 7,
           "AI machine learning research fellowship 2026 fully funded international no PhD required"),
]


def load_whitelist() -> List[Source]:
    """Return all whitelisted sources, highest quality first."""
    return sorted(SOURCES, key=lambda s: s.quality, reverse=True)


def domain_quality(url: str) -> int:
    """Best-effort quality score for a result URL based on its domain."""
    low = (url or "").lower()
    best = 0
    for s in SOURCES:
        if s.domain and s.domain in low:
            best = max(best, s.quality)
    return best
