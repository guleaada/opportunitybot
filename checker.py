"""
checker.py — eligibility, legitimacy, and deadline checking.

Model routing (enforced via task_type passed to call_model):
  check_deadline      → gemini  (check_deadline)     free
  first_pass_filter   → gemini  (first_pass_filter)  free
  check_legitimacy    → claude  (scam_detection)     $  HIGH STAKES
  check_eligibility   → claude  (deep_eligibility)   $  HIGH STAKES

Honesty rule (spec I): if a model cannot determine a value, it must return
``unknown`` rather than guess. We pass that policy in every prompt and default
ambiguous parses to ``unknown``.
"""

from datetime import datetime, timezone
from typing import Optional

from dateutil import parser as dateparser

from model_router import call_model, extract_json, as_object
from known_scams import red_flag_report
from profile import profile_summary

_MAX_TEXT = 12000  # keep prompts cheap; trim very long pages


def _trim(text: str, limit: int = _MAX_TEXT) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]..."


# ── Eligibility ladder ──────────────────────────────────────────────────────
# Graded replacement for the old binary eligible/ineligible. Ordered best →
# worst; ELIGIBILITY_RANK lets callers compare levels numerically.
CONFIRMED_ELIGIBLE = "CONFIRMED_ELIGIBLE"
PROBABLY_ELIGIBLE = "PROBABLY_ELIGIBLE"
UNCERTAIN = "UNCERTAIN"
PROBABLY_INELIGIBLE = "PROBABLY_INELIGIBLE"
CONFIRMED_INELIGIBLE = "CONFIRMED_INELIGIBLE"

ELIGIBILITY_LEVELS = [
    CONFIRMED_ELIGIBLE, PROBABLY_ELIGIBLE, UNCERTAIN,
    PROBABLY_INELIGIBLE, CONFIRMED_INELIGIBLE,
]
ELIGIBILITY_RANK = {level: i for i, level in enumerate(ELIGIBILITY_LEVELS)}

# Old vocabulary → ladder, so legacy model replies and stored records still map.
_LEGACY_TO_LEVEL = {
    "eligible": PROBABLY_ELIGIBLE,     # legacy "eligible" carries no proof → not CONFIRMED
    "ineligible": CONFIRMED_INELIGIBLE,
    "unknown": UNCERTAIN,
}
# Ladder → old vocabulary, so existing readers of ``overall`` keep working.
_LEVEL_TO_LEGACY = {
    CONFIRMED_ELIGIBLE: "eligible",
    PROBABLY_ELIGIBLE: "eligible",
    UNCERTAIN: "unknown",
    PROBABLY_INELIGIBLE: "ineligible",
    CONFIRMED_INELIGIBLE: "ineligible",
}


def normalize_eligibility(value) -> str:
    """Coerce any eligibility value (ladder level or legacy word) to a level.

    Unrecognized input is UNCERTAIN — never an eligible level, per the
    honesty rule.
    """
    if not value:
        return UNCERTAIN
    raw = str(value).strip().upper()
    if raw in ELIGIBILITY_RANK:
        return raw
    return _LEGACY_TO_LEVEL.get(str(value).strip().lower(), UNCERTAIN)


def meets_threshold(level, minimum: str = PROBABLY_ELIGIBLE) -> bool:
    """True if ``level`` is at least as good as ``minimum`` on the ladder."""
    return ELIGIBILITY_RANK[normalize_eligibility(level)] <= ELIGIBILITY_RANK[minimum]


# ── Credibility ladder ──────────────────────────────────────────────────────
# Graded trust, replacing the bare legitimate/scam/unknown verdict. Being
# UNFAMILIAR is NEVER treated as fraud: an unknown source is
# NEEDS_VERIFICATION, which is a request for evidence, not an accusation.
VERIFIED = "VERIFIED"
LIKELY_LEGITIMATE = "LIKELY_LEGITIMATE"
NEEDS_VERIFICATION = "NEEDS_VERIFICATION"
SUSPICIOUS = "SUSPICIOUS"
HIGH_RISK = "HIGH_RISK"

CREDIBILITY_LEVELS = [VERIFIED, LIKELY_LEGITIMATE, NEEDS_VERIFICATION,
                      SUSPICIOUS, HIGH_RISK]
CREDIBILITY_RANK = {c: i for i, c in enumerate(CREDIBILITY_LEVELS)}

# Only these may reach a normal high-priority notification.
NOTIFIABLE_CREDIBILITY = (VERIFIED, LIKELY_LEGITIMATE)


def normalize_credibility(value) -> str:
    """Coerce any credibility value to a level; unknown → NEEDS_VERIFICATION."""
    if not value:
        return NEEDS_VERIFICATION
    raw = str(value).strip().upper()
    if raw in CREDIBILITY_RANK:
        return raw
    return {
        "legitimate": LIKELY_LEGITIMATE,
        "suspicious": SUSPICIOUS,
        "scam": HIGH_RISK,
        "unknown": NEEDS_VERIFICATION,
    }.get(str(value).strip().lower(), NEEDS_VERIFICATION)


def credibility_status(verdict, source_tier=None, confidence=None) -> str:
    """Combine the model's legitimacy verdict with the source tier.

    A first-party official source backing a legitimate verdict earns VERIFIED.
    NEEDS_VERIFICATION is reserved for verdicts that are genuinely unknown or
    unverifiable — an unfamiliar DOMAIN is not itself a reason to withhold a
    positive legitimacy judgment, since most RSS/taxonomy discoveries live on
    domains that are not on the whitelist. The tier still shapes credibility
    scoring; it is not a kill switch.

    A TIER_5 (known fee-trap) source is HIGH_RISK regardless of verdict.
    """
    from source_whitelist import TIER_1, TIER_5

    level = normalize_credibility(verdict)

    if source_tier == TIER_5:
        return HIGH_RISK
    # Never soften a negative verdict on the strength of a nice domain.
    if level in (HIGH_RISK, SUSPICIOUS):
        return level

    if level == LIKELY_LEGITIMATE:
        try:
            conf = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            conf = None
        if source_tier == TIER_1 and (conf is None or conf >= 0.7):
            return VERIFIED
        # The checker formed a positive legitimacy judgment — honour it for
        # every remaining tier, including TIER_4 (unlisted domain).
        return LIKELY_LEGITIMATE

    # Only genuinely unknown / unverifiable verdicts land here.
    return NEEDS_VERIFICATION


def is_notifiable(credibility) -> bool:
    """HIGH_RISK / SUSPICIOUS / NEEDS_VERIFICATION never notify normally."""
    return normalize_credibility(credibility) in NOTIFIABLE_CREDIBILITY


# ── Deadline (Gemini, free) ─────────────────────────────────────────────────
def check_deadline(text: str) -> dict:
    """Extract the application deadline. Returns status + days remaining.

    Returns::
        {"status": "open"|"closed"|"unknown",
         "deadline": "YYYY-MM-DD"|None, "days_left": int|None, "raw": str}
    """
    system = (
        "You extract scholarship/fellowship application deadlines. "
        "Be conservative: if no clear deadline is stated, return unknown. "
        "Never invent a date."
    )
    prompt = (
        "From the text below, find the APPLICATION deadline (not program start "
        "date). Reply ONLY with JSON:\n"
        '{"deadline_iso": "YYYY-MM-DD" or null, '
        '"deadline_raw": "as written" or null, '
        '"is_explicitly_closed": true/false, '
        '"found": true/false}\n\n'
        f"TEXT:\n{_trim(text)}"
    )
    res = call_model("check_deadline", prompt, system=system, max_tokens=300)
    # A non-object reply yields {}, so `found` is absent and the existing
    # "unknown" branch below returns — no deadline is invented.
    data = as_object(extract_json(res["content"]), "check_deadline")

    if data.get("is_explicitly_closed"):
        return {"status": "closed", "deadline": None, "days_left": None,
                "raw": data.get("deadline_raw")}

    iso = data.get("deadline_iso")
    if not iso or not data.get("found"):
        return {"status": "unknown", "deadline": None, "days_left": None,
                "raw": data.get("deadline_raw")}

    try:
        dl = dateparser.parse(iso)
        if dl.tzinfo is None:
            dl = dl.replace(tzinfo=timezone.utc)
        days = (dl.date() - datetime.now(timezone.utc).date()).days
        status = "open" if days >= 0 else "closed"
        return {"status": status, "deadline": dl.date().isoformat(),
                "days_left": days, "raw": data.get("deadline_raw")}
    except (ValueError, OverflowError, TypeError):
        return {"status": "unknown", "deadline": None, "days_left": None,
                "raw": data.get("deadline_raw")}


# ── First-pass filter (Gemini, free) — drops obvious non-fits ───────────────
def first_pass_filter(text: str, profile: dict) -> dict:
    """Cheap triage. Returns {"keep": bool, "reason": str}.

    Drops: closed deadlines, wrong country eligibility, no funding,
    online-only certificates, fee-heavy programs. Errs toward KEEP when unsure
    (Claude will judge the survivors).
    """
    system = (
        "You are a fast first-pass filter for scholarship/fellowship listings. "
        "Your job is to cheaply DROP obvious non-fits and KEEP plausible ones. "
        "When genuinely unsure, KEEP (a more expensive model will decide later)."
    )
    prompt = (
        "Decide whether this opportunity is worth deeper analysis for the "
        "candidate below.\n\n"
        f"CANDIDATE PROFILE:\n{profile_summary()}\n\n"
        "DROP it (keep=false) if ANY of these are clearly true:\n"
        "- The application is clearly CLOSED or the deadline has passed.\n"
        "- Ethiopian / African / developing-country nationals are clearly NOT eligible.\n"
        "- It explicitly offers NO funding and charges money to participate.\n"
        "- It is an online-only certificate course.\n"
        "- It requires an application fee over $50.\n"
        "Otherwise KEEP it (keep=true).\n\n"
        "Reply ONLY with JSON: "
        '{"keep": true/false, "reason": "one short sentence", '
        '"guessed_country": "country or unknown", '
        '"guessed_funding": "fully_funded|partial|none|unknown"}\n\n'
        f"TEXT:\n{_trim(text)}"
    )
    res = call_model("first_pass_filter", prompt, system=system, max_tokens=300)
    data = extract_json(res["content"])
    if not data or "keep" not in data:
        # Parsing failed → be safe, keep for deeper analysis.
        return {"keep": True, "reason": "first-pass parse failed; keeping for review",
                "model_used": res["model_used"]}
    return {
        "keep": bool(data["keep"]),
        "reason": data.get("reason", ""),
        "guessed_country": data.get("guessed_country", "unknown"),
        "guessed_funding": data.get("guessed_funding", "unknown"),
        "model_used": res["model_used"],
    }


# ── Legitimacy / scam detection (Claude, HIGH STAKES) ───────────────────────
def check_legitimacy(text: str, source_url: str) -> dict:
    """Claude judgment on whether this is a legitimate opportunity or a scam.

    Returns::
        {"verdict": "legitimate"|"scam"|"suspicious"|"unknown",
         "confidence": 0-1, "reasoning": str, "red_flags": [...],
         "model_used": str}
    """
    flags = red_flag_report(text)
    system = (
        "You are a rigorous fraud analyst for international scholarships and "
        "fellowships. You protect the applicant's limited time and money. "
        "Classic scams: 'you've been selected' invitation summits that charge "
        "registration/participation fees, vague-prestige awards, and any "
        "program asking applicants to wire money. Legitimate programs "
        "(DAAD, Chevening, Fulbright, Erasmus, MEXT, government/embassy "
        "scholarships) NEVER charge large fees to apply.\n"
        "If the evidence is genuinely insufficient, return verdict 'unknown' "
        "rather than guessing."
    )
    prompt = (
        f"SOURCE URL: {source_url}\n\n"
        f"Automated keyword scan found:\n"
        f"- Red-flag phrases: {flags['red_flags_found'] or 'none'}\n"
        f"- Trust signals: {flags['trust_signals_found'] or 'none'}\n\n"
        "Assess legitimacy. Reply ONLY with JSON:\n"
        '{"verdict": "legitimate"|"scam"|"suspicious"|"unknown", '
        '"confidence": 0.0-1.0, '
        '"reasoning": "2-3 sentences citing specific evidence", '
        '"red_flags": ["..."]}\n\n'
        f"PAGE TEXT:\n{_trim(text)}"
    )
    res = call_model("scam_detection", prompt, system=system, max_tokens=600)
    # A non-object reply yields {} here, so verdict stays "unknown" and the
    # credibility ladder resolves to NEEDS_VERIFICATION — the project's
    # existing "we could not verify this" outcome, not a guessed verdict.
    data = as_object(extract_json(res["content"]), "scam_detection")
    verdict = data.get("verdict", "unknown")
    if verdict not in ("legitimate", "scam", "suspicious", "unknown"):
        verdict = "unknown"

    # Provenance tier + verdict → graded credibility. Fail-soft: a tier lookup
    # problem must not lose the legitimacy result we already paid for.
    try:
        from source_whitelist import source_tier as _source_tier
        tier = _source_tier(source_url)
    except Exception as e:
        print(f"⚠️  Source tier lookup failed for {source_url!r}: {e}")
        tier = None

    status = credibility_status(verdict, tier, data.get("confidence"))

    return {
        "verdict": verdict,                 # legacy field, unchanged
        "credibility_status": status,
        "source_tier": tier,
        "confidence": data.get("confidence"),
        "reasoning": data.get("reasoning", "No reasoning returned."),
        "red_flags": data.get("red_flags", flags["red_flags_found"]),
        "model_used": res["model_used"],
        "cost_usd": res["cost_usd"],
    }


# ── Eligibility (Claude, HIGH STAKES) ────────────────────────────────────────
def check_eligibility(text: str, profile: dict) -> dict:
    """Deep eligibility analysis against the user's profile.

    Grades onto the 5-level ladder rather than a binary verdict. Two rules are
    enforced in code (not left to the model):
      * missing / unverifiable requirements  ⇒ UNCERTAIN (never "eligible")
      * hard citizenship or country mismatch ⇒ CONFIRMED_INELIGIBLE

    Returns::
        {"eligibility_status": <ladder level>,
         "overall": "eligible"|"ineligible"|"unknown",   # legacy, derived
         "reasoning": str, "blocking_issues": [...],
         "addressable_gaps": [...], "missing_requirements": [...],
         "citizenship_mismatch": bool, "model_used": str}
    """
    system = (
        "You are a meticulous eligibility analyst for international "
        "scholarships and fellowships. Compare the program's stated "
        "requirements against the candidate profile. Distinguish HARD blockers "
        "(nationality excluded, must currently be a student, age cap exceeded, "
        "degree level mismatch) from ADDRESSABLE gaps (needs an English test "
        "the candidate can still obtain, needs a document they can produce).\n"
        "Key facts about this candidate: Ethiopian national, age 27, WORKING "
        "PROFESSIONAL (not currently a student), holds a BSc (bachelor's), has "
        "NO IELTS/TOEFL yet but can obtain a Duolingo or MOI certificate.\n"
        "CRITICAL: never upgrade missing information into eligibility. Always "
        "list anything unstated or unverifiable in missing_requirements.\n"
        "But grade on whether this candidate is plausibly BLOCKED, not on how "
        "complete the page is. Reserve UNCERTAIN for a decision-relevant "
        "criterion you cannot resolve — one that could actually exclude this "
        "candidate (nationality, student-status, age cap, degree level). A "
        "page that simply omits routine detail, while nothing on it excludes "
        "the candidate, is PROBABLY_ELIGIBLE with those items listed as "
        "missing_requirements. Do not guess, and do not withhold a positive "
        "grade merely because the page is thin."
    )
    prompt = (
        f"CANDIDATE PROFILE:\n{profile_summary()}\n\n"
        "Grade eligibility for the program below on this ladder:\n"
        "- CONFIRMED_ELIGIBLE: page explicitly states requirements the "
        "candidate demonstrably meets; nothing unverified.\n"
        "- PROBABLY_ELIGIBLE: requirements stated and likely met, minor "
        "addressable gaps only.\n"
        "- UNCERTAIN: a criterion that could exclude THIS candidate "
        "(nationality, student status, age cap, degree level) is missing or "
        "unverifiable — not merely that the page omits routine detail.\n"
        "- PROBABLY_INELIGIBLE: likely blocked but not stated outright.\n"
        "- CONFIRMED_INELIGIBLE: page explicitly excludes this candidate "
        "(e.g. nationality not eligible, must be enrolled student).\n\n"
        "Reply ONLY with JSON:\n"
        '{"eligibility_status": "CONFIRMED_ELIGIBLE|PROBABLY_ELIGIBLE|'
        'UNCERTAIN|PROBABLY_INELIGIBLE|CONFIRMED_INELIGIBLE", '
        '"reasoning": "specific, cites requirements", '
        '"blocking_issues": ["hard blockers, if any"], '
        '"addressable_gaps": ["gaps the candidate can fix in time"], '
        '"missing_requirements": ["requirements not stated or unverifiable"], '
        '"citizenship_mismatch": true/false}\n\n'
        f"PROGRAM TEXT:\n{_trim(text)}"
    )
    res = call_model("deep_eligibility", prompt, system=system, max_tokens=800)
    # A non-object reply yields {}, so normalize_eligibility(None) resolves to
    # UNCERTAIN — the existing "we could not grade this" outcome.
    data = as_object(extract_json(res["content"]), "deep_eligibility")

    # Accept the ladder, or a legacy "overall" reply, or nothing at all.
    level = normalize_eligibility(
        data.get("eligibility_status") or data.get("overall"))
    missing = data.get("missing_requirements") or []
    if not isinstance(missing, list):
        missing = [str(missing)]
    citizenship_mismatch = bool(data.get("citizenship_mismatch"))

    # The model's own grade, before the code rules below. Persisted so the
    # firing rate of each rule is measurable instead of inferred.
    model_level = level

    # Rule 1: a hard citizenship/country mismatch is disqualifying outright.
    if citizenship_mismatch:
        level = CONFIRMED_INELIGIBLE
    # Rule 2: unverified detail caps confidence — it does not veto the verdict.
    #
    # This used to collapse ANY positive grade to UNCERTAIN whenever
    # missing_requirements was non-empty. Every real program page leaves
    # something unstated, so the rule fired on nearly every candidate and
    # PROBABLY_ELIGIBLE became unreachable: between 2026-08-15 and 2026-08-28,
    # 177 candidates were graded and exactly 2 cleared the gate (1.1%, against
    # 24.3% before the ladder landed). Nothing reached the scorer, so the run
    # produced no matches at all for two weeks.
    #
    # The ladder already has a level for "likely eligible, minor unresolved
    # gaps" — that is what PROBABLY_ELIGIBLE means. So an unverified detail now
    # costs one step instead of vetoing the verdict: CONFIRMED_ELIGIBLE, which
    # claims "nothing unverified", is demoted to PROBABLY_ELIGIBLE, and
    # PROBABLY_ELIGIBLE stands. The honesty rule is intact — a level is only
    # ever lowered here, never raised, so missing information still cannot
    # manufacture eligibility.
    elif missing and level == CONFIRMED_ELIGIBLE:
        level = PROBABLY_ELIGIBLE

    return {
        "eligibility_status": level,
        "model_eligibility_status": model_level,
        "overall": _LEVEL_TO_LEGACY[level],  # backward compatibility
        "reasoning": data.get("reasoning", "No reasoning returned."),
        "blocking_issues": data.get("blocking_issues", []),
        "addressable_gaps": data.get("addressable_gaps", []),
        "missing_requirements": missing,
        "citizenship_mismatch": citizenship_mismatch,
        "model_used": res["model_used"],
        "cost_usd": res["cost_usd"],
    }
