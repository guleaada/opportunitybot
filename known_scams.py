"""
known_scams.py — pre-populated scam / fee-trap database + heuristics.

Two layers:
1. KNOWN_SCAMS         — exact-ish name/domain blocklist (hard block, no model).
2. RED_FLAG_KEYWORDS   — heuristic phrases that *raise suspicion* (not an
                         automatic block; they feed the Claude scam check).

The classic pattern these target: "you've been selected" invitation summits
that charge a large registration/participation fee, promise vague "global
leadership" prestige, and provide no real funding.
"""

# Hard-block: if the opportunity name or URL matches any of these, drop it
# before spending a single token. Matching is case-insensitive substring.
KNOWN_SCAMS = [
    # Acronym fee-summits called out by the user. Matched via their full
    # descriptive phrase only — bare 3-5 letter acronyms (e.g. "GBS" is also
    # a bank/business-school abbreviation) are too collision-prone to
    # hard-block without a model review, so they are excluded here.
    {"name": "CSCD", "aliases": ["civil society for", "cscd summit"], "domains": []},
    {"name": "CGDL", "aliases": ["global development leadership"], "domains": []},
    {"name": "GBS", "aliases": ["global business summit"], "domains": []},
    {"name": "ICCSL", "aliases": ["international conference on civil society"], "domains": []},
    # Generic vague-prestige summit names.
    {"name": "Global Business Summit", "aliases": ["global business summit"], "domains": []},
    {"name": "International Leadership Summit", "aliases": ["international leadership summit"], "domains": []},
    {"name": "World Leadership Congress", "aliases": ["world leadership congress"], "domains": []},
    {"name": "Global Youth Leadership Summit", "aliases": ["global youth leadership summit"], "domains": []},
    {"name": "International Youth Conference (fee-based)", "aliases": ["international youth conference"], "domains": []},
    {"name": "World Peace Summit (invitation fee)", "aliases": ["world peace summit"], "domains": []},
    {"name": "Global Leaders Forum", "aliases": ["global leaders forum"], "domains": []},
    {"name": "International Excellence Awards", "aliases": ["international excellence award"], "domains": []},
    {"name": "Global Conference on Career", "aliases": ["global conference on career"], "domains": []},
]

# Phrases that commonly appear in fee-trap / scam invitations. Presence does
# NOT auto-block — it's signal handed to the Claude scam-detection prompt.
RED_FLAG_KEYWORDS = [
    "you have been selected",
    "you are cordially invited",
    "registration fee",
    "participation fee",
    "delegate fee",
    "processing fee",
    "nomination fee",
    "membership fee",
    "secure your seat",
    "limited slots available",
    "pay via western union",
    "pay via moneygram",
    "send payment to",
    "wire transfer to secure",
    "honorary award",
    "lifetime achievement award",
    "prestigious recognition",
    "your profile was shortlisted",
    "congratulations you qualify",
    "act now",
    "non-refundable fee",
    "bank charges apply",
    "visa processing fee payable",
    "exclusive invitation",
    "early bird discount",
    "vip package",
    "venue confirmation upon payment",
    "limited seats",
    "early bird registration",
]

# Phrases that *increase trust* (presence of these counterbalances red flags).
TRUST_SIGNALS = [
    "fully funded",
    "no application fee",
    "government scholarship",
    "ministry of",
    "embassy of",
    "official scholarship",
    "deutscher akademischer austauschdienst",  # DAAD full name
    "foreign and commonwealth office",  # Chevening
    "fulbright",
    "erasmus",
]


def check_known_scam(name: str, url: str = "") -> dict:
    """Hard lookup against the scam blocklist. No model call.

    Returns ``{"is_scam": bool, "matched": str | None, "reason": str}``.
    """
    haystack = f"{name or ''} {url or ''}".lower()
    for entry in KNOWN_SCAMS:
        for alias in entry["aliases"]:
            alias = alias.lower()
            if alias and alias in haystack:
                return {
                    "is_scam": True,
                    "matched": entry["name"],
                    "reason": f"Matches known fee-trap pattern: {entry['name']}",
                }
        for domain in entry.get("domains", []):
            if domain and domain.lower() in haystack:
                return {
                    "is_scam": True,
                    "matched": entry["name"],
                    "reason": f"Matches known scam domain for {entry['name']}",
                }
    return {"is_scam": False, "matched": None, "reason": ""}


def red_flag_report(text: str) -> dict:
    """Count red-flag and trust phrases in ``text`` for the scam-check prompt."""
    low = (text or "").lower()
    flags = [kw for kw in RED_FLAG_KEYWORDS if kw in low]
    trust = [kw for kw in TRUST_SIGNALS if kw in low]
    return {
        "red_flags_found": flags,
        "trust_signals_found": trust,
        "red_flag_count": len(flags),
        "trust_signal_count": len(trust),
    }
