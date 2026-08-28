"""
taxonomy.py — opportunity categories + search-query families.

Widens discovery beyond scholarships/fellowships. ``CATEGORIES`` is a plain
dict of ``category -> [query template, ...]`` so adding a new category (or a
new query for an existing one) is a one-line edit with no code changes
anywhere else.

The queries generated here are appended to the existing whitelist queries in
main.py's ``run_scan`` — they feed the SAME CSE/RSS/scoring pipeline, not a
parallel one. Nothing here calls a model or the network.
"""

from datetime import datetime, timezone
from typing import Dict, List, Tuple

# Category -> 1-3 search query templates.
# Plain data: append a category or a query string and it is picked up
# automatically by all_search_queries().
CATEGORIES: Dict[str, List[str]] = {
    # ── Funding & study ────────────────────────────────────────────────────
    # NOTE ON REGISTER — read before editing these.
    # The profile is a WORKING PROFESSIONAL and solo founder, not a student.
    # These queries were originally written in student-scholarship register
    # ("international students", "masters scholarship developing countries"),
    # which is why discovery kept returning enrolment-gated programmes that
    # the eligibility gate then correctly rejected: 28 of the 160 stored
    # rejections read "must be currently enrolled". Keep this section in
    # professional / founder / practitioner vocabulary.
    "grants": [
        "innovation grant open call founders Global South apply",
        "individual grant funding social entrepreneur Africa apply",
    ],
    "scholarships": [
        "fully funded masters scholarship working professionals apply",
        "scholarship for applicants with work experience developing countries",
    ],
    "fellowships": [
        "fully funded fellowship early career professional no degree required",
        "professional fellowship Africa mid-career practitioner apply",
        "AI technology fellowship stipend non-academic applicants apply",
    ],
    "programs": [
        "leadership program fully funded professionals apply",
        "training program sponsored international participants",
    ],
    "research": [
        "research fellowship independent researcher no PhD required funded",
        "visiting practitioner program funded apply international",
    ],

    # ── Founder / builder funding (this profile's strongest fit) ───────────
    # SebilAI is an AI crop-disease platform for Ethiopian smallholders, so
    # agritech, climate-adaptation and AI-for-good funders are populations
    # this profile sits squarely inside — and they rarely gate on enrolment,
    # nationality or IELTS, which are the three constraints killing most of
    # the current funnel (28 enrolment, 40 nationality, 11 English cert).
    "agritech_climate": [
        "agritech innovation grant smallholder farmers funding apply",
        "climate adaptation agriculture technology grant Africa founders",
        "food security innovation challenge funding apply Africa",
    ],
    "ai_for_good": [
        "AI for social good grant funding developing countries apply",
        "AI for agriculture health impact funding open call apply",
    ],
    "social_entrepreneurship": [
        "social entrepreneur fellowship funding Africa apply",
        "social impact venture grant non-equity founders apply",
    ],
    "founder_fellowships": [
        "founder fellowship equity free funding solo founder apply",
        "entrepreneurship fellowship early stage founder stipend apply",
    ],

    # ── Work & income ──────────────────────────────────────────────────────
    "remote_jobs": [
        "remote job worldwide hiring apply software",
        "fully remote position hire globally no relocation",
    ],
    "dev_jobs": [
        "software engineer remote job international applicants",
        "backend developer remote hiring visa sponsorship",
    ],
    "freelance": [
        "freelance contract remote developer paid gig",
        "paid freelance project software worldwide",
    ],
    "paid_internships": [
        "paid internship remote international students apply",
        "software engineering internship stipend international",
    ],

    # ── Competitions & prizes ──────────────────────────────────────────────
    "hackathons": [
        "hackathon online prize money register",
        "global hackathon cash prize open worldwide",
    ],
    "coding_competitions": [
        "coding competition prize money online register",
        "programming contest cash prize international",
    ],
    "startup_competitions": [
        "startup competition prize funding apply Africa",
        "pitch competition equity free grant apply",
    ],
    "cash_prize_competitions": [
        "competition cash prize open international apply",
        "award competition prize money open call",
    ],
    "ai_competitions": [
        "AI machine learning competition prize money kaggle",
        "AI challenge prize open worldwide apply",
    ],

    # ── Bounties ───────────────────────────────────────────────────────────
    "open_source_bounties": [
        "open source bounty paid issue",
        "github bounty program paid contribution",
        "paid open source contribution program",
    ],
    "security_bounties": [
        "bug bounty program rewards researchers",
        "vulnerability disclosure program paid bounty",
    ],

    # ── Startup / builder ecosystem ────────────────────────────────────────
    "accelerators": [
        "startup accelerator program apply funding Africa",
        "remote accelerator batch applications open",
    ],
    "incubators": [
        "startup incubator program apply funding early stage",
        "incubator open call founders apply",
    ],
    "startup_funding": [
        "pre-seed funding open application founders Africa",
        "micro grant founders no equity apply",
    ],

    # ── Platform / creator programs ────────────────────────────────────────
    "creator_programs": [
        "creator program paid technical writing apply",
        "developer advocate ambassador program paid",
    ],
    "developer_programs": [
        "developer program credits sponsorship apply",
        "cloud credits program startups developers apply",
    ],
}


def _year_hint() -> str:
    """Current + next year, so generated queries stay fresh over time."""
    y = datetime.now(timezone.utc).year
    return f"{y} {y + 1}"


def all_categories() -> List[str]:
    """Every category name, sorted for stable ordering."""
    return sorted(CATEGORIES)


def queries_for(category: str, with_year: bool = True) -> List[str]:
    """Search queries for one category. Unknown category → empty list."""
    hint = f" {_year_hint()}" if with_year else ""
    return [f"{q}{hint}" for q in CATEGORIES.get(category, [])]


def all_search_queries(with_year: bool = True) -> List[Tuple[str, str]]:
    """Flat ``[(category, query), ...]`` across every category."""
    out: List[Tuple[str, str]] = []
    for category in all_categories():
        for query in queries_for(category, with_year=with_year):
            out.append((category, query))
    return out


def query_count() -> int:
    """Total number of generated queries across all categories."""
    return sum(len(v) for v in CATEGORIES.values())
