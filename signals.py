"""
signals.py — semantic "hidden opportunity" detection.

Catches opportunities whose pages never say "scholarship", "grant" or "job":
prize pools, bounties, open calls, paid challenges, startup support. Cheap by
design:

  1. phrase scan (free, no model) settles the clear cases;
  2. only genuinely ambiguous pages go to the CHEAP model (Gemini/Groq via the
     existing router) — never Claude;
  3. a per-scan classification budget bounds even the free calls.

Every path fails soft: a model error, a rate limit or a blocked source logs a
warning and falls back to the phrase-scan verdict. Nothing here can crash a scan.
"""

import os

from model_router import call_model, extract_json

# Free-tier task type; see TASK_ROUTING in model_router.py.
CLASSIFY_TASK = "classify_opportunity"

# Decisive: if the page says one of these, it is an opportunity.
STRONG_SIGNALS = [
    "applications open", "application is open", "call for applications",
    "now accepting applications", "open call", "apply now",
    "prize pool", "cash prize", "prize money", "winners receive",
    "bounty", "bug bounty", "paid challenge",
    "funding available", "grant funding", "equity-free",
    "selected participants receive", "selected candidates will receive",
    "travel and accommodation provided", "all expenses paid",
    "fully funded", "stipend provided", "monthly stipend",
    "developers invited to submit", "request for proposals",
    "startup support", "accelerator program", "incubator program",
    "submissions are open", "register to compete",
]

# Suggestive but not conclusive — two or more of these trigger a cheap
# model check rather than an outright decision.
WEAK_SIGNALS = [
    "deadline", "eligibility", "apply by", "submit your", "opportunity",
    "programme", "program", "fellowship", "residency", "competition",
    "challenge", "hackathon", "award", "stipend", "sponsored",
    "no application fee", "free to enter", "mentorship", "cohort",
    "reward", "compensation", "remote", "invited to apply",
]

# Pages that merely *discuss* opportunities rather than offering one.
NEGATIVE_SIGNALS = [
    "this article", "blog post", "privacy policy", "terms of service",
    "cookie policy", "page not found", "404", "subscribe to our newsletter",
    "advertisement", "sponsored content",
]

# Bound even the free classification calls per scan.
_MAX_CLASSIFICATIONS = int(os.getenv("MAX_OPPORTUNITY_CLASSIFICATIONS", "40"))
_classifications_used = 0


def reset_classification_budget() -> None:
    """Called at the start of a scan so the budget is per-run."""
    global _classifications_used
    _classifications_used = 0


def classifications_used() -> int:
    return _classifications_used


def signal_report(text: str, title: str = "") -> dict:
    """Pure phrase scan — no model, no network."""
    low = f"{title or ''} {text or ''}".lower()
    strong = [s for s in STRONG_SIGNALS if s in low]
    weak = [s for s in WEAK_SIGNALS if s in low]
    negative = [s for s in NEGATIVE_SIGNALS if s in low]
    return {
        "strong": strong, "weak": weak, "negative": negative,
        "strong_count": len(strong), "weak_count": len(weak),
        "negative_count": len(negative),
    }


def _classify_with_cheap_model(text: str, title: str, report: dict) -> dict:
    """Ambiguous case → free model. Never Claude, never fatal."""
    global _classifications_used
    if _classifications_used >= _MAX_CLASSIFICATIONS:
        return {"is_opportunity": False, "confidence": "low",
                "method": "budget_exhausted",
                "reason": "classification budget for this scan is used up"}

    system = (
        "You decide whether a web page is offering a real OPPORTUNITY a person "
        "can apply to or compete for — a grant, scholarship, fellowship, job, "
        "paid internship, bounty, hackathon, competition, accelerator or "
        "funding call. A page that merely discusses or lists such things "
        "without offering one is NOT an opportunity. Answer honestly; if you "
        "cannot tell, say unsure."
    )
    prompt = (
        f"TITLE: {title}\n"
        f"Phrase scan found — strong: {report['strong'] or 'none'}; "
        f"weak: {report['weak'] or 'none'}\n\n"
        "Reply ONLY with JSON:\n"
        '{"is_opportunity": true/false/"unsure", '
        '"opportunity_type": "grant|scholarship|fellowship|job|internship|'
        'bounty|hackathon|competition|accelerator|funding|other|none", '
        '"reason": "one short sentence"}\n\n'
        f"PAGE TEXT:\n{(text or '')[:4000]}"
    )
    try:
        _classifications_used += 1
        res = call_model(CLASSIFY_TASK, prompt, system=system, max_tokens=200)
        data = extract_json(res["content"]) or {}
        verdict = data.get("is_opportunity")
        if isinstance(verdict, str):
            # "unsure" (or anything non-boolean) must not become a yes.
            is_opp = verdict.strip().lower() == "true"
            confidence = "low"
        else:
            is_opp = bool(verdict)
            confidence = "medium"
        return {
            "is_opportunity": is_opp,
            "confidence": confidence,
            "method": "cheap_model",
            "opportunity_type": data.get("opportunity_type", "unknown"),
            "reason": data.get("reason", ""),
            "model_used": res.get("model_used"),
        }
    except Exception as e:  # rate limit, network, provider outage — fail soft
        print(f"⚠️  Opportunity classification failed ({e}) — "
              f"falling back to phrase scan.")
        return {"is_opportunity": report["weak_count"] >= 3,
                "confidence": "low", "method": "signal_fallback",
                "reason": f"classifier unavailable: {e}"}


def detect_opportunity(text: str, title: str = "", allow_model: bool = True) -> dict:
    """Is this page an actual opportunity, even if it never says so?

    Returns ``{"is_opportunity", "confidence", "method", "signals", ...}``.
    Never raises.
    """
    try:
        report = signal_report(text, title)

        # Decisive phrase → done, no model spend.
        if report["strong_count"] >= 1:
            return {"is_opportunity": True, "confidence": "high",
                    "method": "signal", "signals": report,
                    "reason": f"strong signal: {report['strong'][0]}"}

        # Suggestive only → ask the cheap model.
        if report["weak_count"] >= 2 and allow_model:
            out = _classify_with_cheap_model(text, title, report)
            out["signals"] = report
            return out

        return {"is_opportunity": False, "confidence": "low",
                "method": "signal", "signals": report,
                "reason": "no opportunity signals found"}
    except Exception as e:  # detection must never break a scan
        print(f"⚠️  Opportunity detection failed ({e}) — treating as unknown.")
        return {"is_opportunity": False, "confidence": "low",
                "method": "error", "signals": {}, "reason": str(e)}
