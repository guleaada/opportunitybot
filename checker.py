"""
checker.py — eligibility, legitimacy, and deadline checking.

Model routing (enforced via task_type passed to call_model):
  check_deadline      → gemini  (first_pass_filter)  free
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

from model_router import call_model, extract_json
from known_scams import red_flag_report
from profile import profile_summary

_MAX_TEXT = 12000  # keep prompts cheap; trim very long pages


def _trim(text: str, limit: int = _MAX_TEXT) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]..."


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
    res = call_model("first_pass_filter", prompt, system=system, max_tokens=300)
    data = extract_json(res["content"]) or {}

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
    except (ValueError, OverflowError):
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
    data = extract_json(res["content"]) or {}
    verdict = data.get("verdict", "unknown")
    if verdict not in ("legitimate", "scam", "suspicious", "unknown"):
        verdict = "unknown"
    return {
        "verdict": verdict,
        "confidence": data.get("confidence"),
        "reasoning": data.get("reasoning", "No reasoning returned."),
        "red_flags": data.get("red_flags", flags["red_flags_found"]),
        "model_used": res["model_used"],
        "cost_usd": res["cost_usd"],
    }


# ── Eligibility (Claude, HIGH STAKES) ────────────────────────────────────────
def check_eligibility(text: str, profile: dict) -> dict:
    """Deep eligibility analysis against the user's profile.

    Returns::
        {"overall": "eligible"|"ineligible"|"unknown",
         "reasoning": str, "blocking_issues": [...],
         "addressable_gaps": [...], "model_used": str}
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
        "If requirements are unclear or missing, return 'unknown' — do not guess."
    )
    prompt = (
        f"CANDIDATE PROFILE:\n{profile_summary()}\n\n"
        "Analyze eligibility for the program described below. Reply ONLY with "
        "JSON:\n"
        '{"overall": "eligible"|"ineligible"|"unknown", '
        '"reasoning": "specific, cites requirements", '
        '"blocking_issues": ["hard blockers, if any"], '
        '"addressable_gaps": ["gaps the candidate can fix in time"]}\n\n'
        f"PROGRAM TEXT:\n{_trim(text)}"
    )
    res = call_model("deep_eligibility", prompt, system=system, max_tokens=800)
    data = extract_json(res["content"]) or {}
    overall = data.get("overall", "unknown")
    if overall not in ("eligible", "ineligible", "unknown"):
        overall = "unknown"
    return {
        "overall": overall,
        "reasoning": data.get("reasoning", "No reasoning returned."),
        "blocking_issues": data.get("blocking_issues", []),
        "addressable_gaps": data.get("addressable_gaps", []),
        "model_used": res["model_used"],
        "cost_usd": res["cost_usd"],
    }
