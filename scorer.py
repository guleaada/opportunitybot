"""
scorer.py — final opportunity scoring + reasoning (Claude, HIGH STAKES).

Produces the 0-10 score that gates notification. Only ever called on
candidates that already passed first-pass filter + legitimacy + eligibility,
so this is the most expensive and most decision-critical Claude call.
"""

from model_router import call_model, extract_json
from profile import profile_summary

_MAX_TEXT = 12000


def _trim(text: str) -> str:
    text = text or ""
    return text if len(text) <= _MAX_TEXT else text[:_MAX_TEXT] + "\n...[truncated]..."


SCORING_RUBRIC = """\
Score 0-10 (one decimal ok) using this rubric:
- Fit with profile & goals (0-3): destination preference, program type match,
  AI/tech relevance, fellowship vs degree fit.
- Funding quality (0-3): fully funded scores high; partial requiring >$5000
  out of pocket scores low; unfunded scores ~0.
- Winnability for THIS candidate (0-2): realistic odds given BSc Psychology,
  working professional, no IELTS yet, strong AI portfolio.
- Effort vs reward (0-1): lower effort for high reward scores higher.
- Time available (0-1): comfortable deadline scores higher than a near one.
Eligibility ladder: CONFIRMED_ELIGIBLE may score full winnability;
PROBABLY_ELIGIBLE scores slightly lower. UNCERTAIN or worse must NOT score >= 7
— unverified eligibility is not a recommendation to apply.
A score >= 7 means: notify the user, it's worth applying.
"""


def score_opportunity(data: dict, profile: dict) -> dict:
    """Final scoring with Claude.

    ``data`` should include: raw_text, url, and the prior analyses
    (legitimacy, eligibility, documents, complexity, deadline).

    Returns::
        {"overall_score": float, "reasoning": str, "breakdown": {...},
         "funding": str, "recommendation": str, "model_used": str}
    """
    system = (
        "You are the final decision-maker for which scholarships/fellowships "
        "are worth this candidate's limited application time. Be honest and "
        "calibrated — a 7+ is a genuine recommendation to apply. Reserve 9-10 "
        "for excellent, fully-funded, clearly-eligible, high-fit matches."
    )
    prior = {
        "legitimacy": data.get("legitimacy"),
        "eligibility_status": (data.get("eligibility") or {}).get(
            "eligibility_status", "UNCERTAIN"),
        "eligibility": data.get("eligibility"),
        "documents": data.get("documents"),
        "complexity": data.get("complexity"),
        "deadline": data.get("deadline"),
    }
    prompt = (
        f"CANDIDATE PROFILE:\n{profile_summary()}\n\n"
        f"{SCORING_RUBRIC}\n\n"
        f"PRIOR ANALYSIS (already done by other models):\n{prior}\n\n"
        f"OPPORTUNITY URL: {data.get('url')}\n\n"
        "Reply ONLY with JSON:\n"
        '{"overall_score": 0-10, '
        '"breakdown": {"fit": 0-3, "funding": 0-3, "winnability": 0-2, '
        '"effort_vs_reward": 0-1, "time": 0-1}, '
        '"funding": "fully_funded|partial|none|unknown", '
        '"reasoning": "3-4 sentences, specific to this candidate", '
        '"recommendation": "apply|consider|skip"}\n\n'
        f"OPPORTUNITY TEXT:\n{_trim(data.get('raw_text', ''))}"
    )
    res = call_model("final_scoring", prompt, system=system, max_tokens=900)
    parsed = extract_json(res["content"]) or {}

    score = parsed.get("overall_score")
    try:
        score = round(float(score), 1)
    except (TypeError, ValueError):
        score = 0.0

    return {
        "overall_score": score,
        "breakdown": parsed.get("breakdown", {}),
        "funding": parsed.get("funding", "unknown"),
        "reasoning": parsed.get("reasoning", "No reasoning returned."),
        "recommendation": parsed.get("recommendation", "consider"),
        "model_used": res["model_used"],
        "cost_usd": res["cost_usd"],
    }
