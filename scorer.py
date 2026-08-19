"""
scorer.py — final opportunity scoring + reasoning (Claude, HIGH STAKES).

Produces the 0-10 score that gates notification. Only ever called on
candidates that already passed first-pass filter + legitimacy + eligibility,
so this is the most expensive and most decision-critical Claude call.
"""

import math

from model_router import call_model, extract_json, as_object
from checker import (
    normalize_eligibility, meets_threshold, PROBABLY_ELIGIBLE,
    normalize_credibility, is_notifiable,
)
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


# ════════════════════════════════════════════════════════════════════════
# Expected-value engine (pure functions — no model calls, fully testable)
# ════════════════════════════════════════════════════════════════════════
# The eight reported sub-scores, all on the same 0-10 scale as overall_score.
SUB_SCORE_FIELDS = (
    "eligibility_score",      # how firmly the candidate qualifies
    "personal_fit_score",     # match to goals/skills/profile
    "credibility_score",      # is the opportunity real and trustworthy
    "financial_value_score",  # log-compressed size of the reward
    "effort_score",           # 10 = little effort required
    "competition_score",      # 10 = little competition (easy to win)
    "urgency_score",          # 10 = comfortable time to prepare
    "accessibility_score",    # 10 = no visa/location/document barriers
)

# Sub-scores the scoring model may judge. eligibility_score and
# credibility_score are excluded on purpose: they are already decided by
# checker's eligibility ladder and legitimacy verdict.
MODEL_JUDGED_FIELDS = (
    "personal_fit_score", "financial_value_score", "effort_score",
    "competition_score", "urgency_score", "accessibility_score",
)

# Eligibility level → its contribution to probability of success.
_ELIGIBILITY_PROBABILITY = {
    "CONFIRMED_ELIGIBLE": 1.00,
    "PROBABLY_ELIGIBLE": 0.75,
    "UNCERTAIN": 0.30,
    "PROBABLY_INELIGIBLE": 0.08,
    "CONFIRMED_INELIGIBLE": 0.00,
}
# Same ladder expressed on the 0-10 sub-score scale.
_ELIGIBILITY_SCORE = {
    "CONFIRMED_ELIGIBLE": 10.0,
    "PROBABLY_ELIGIBLE": 7.5,
    "UNCERTAIN": 3.0,
    "PROBABLY_INELIGIBLE": 1.0,
    "CONFIRMED_INELIGIBLE": 0.0,
}

# Odds at or above this are treated as "excellent" when normalizing. Realistic
# competitive programs sit far below 1.0, so normalizing against 1.0 would
# crush every genuine opportunity to a near-zero score.
_EXCELLENT_ODDS = 0.50

# Funding label → financial value when no dollar figure is available.
_FUNDING_VALUE_SCORE = {"fully_funded": 8.0, "partial": 4.0, "none": 1.0}

# Credibility ladder → 0-10 sub-score. NEEDS_VERIFICATION sits mid-scale: it
# means "unproven", not "fraudulent".
_CREDIBILITY_SCORE = {
    "VERIFIED": 10.0,
    "LIKELY_LEGITIMATE": 8.0,
    "NEEDS_VERIFICATION": 4.0,
    "SUSPICIOUS": 1.5,
    "HIGH_RISK": 0.0,
}

FINAL_WEIGHTS = {
    "expected_value": 0.40,      # embeds reward x probability
    "personal_fit_score": 0.15,
    "credibility_score": 0.15,
    "effort_score": 0.10,
    "urgency_score": 0.10,
    "accessibility_score": 0.10,
}


def _clamp(x, lo=0.0, hi=10.0):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return None


def reward_score(usd) -> float:
    """Dollar reward → 0-10, log-compressed.

    $100 → 0, $1k → 3.3, $10k → 6.7, $100k → 10. Log scale deliberately: the
    hundredth thousand dollars is worth far less than the first thousand, and
    linear dollars would let one huge prize dominate every ranking.
    """
    try:
        v = float(usd)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return 0.0
    return _clamp((math.log10(max(v, 1.0)) - 2.0) / 3.0 * 10.0)


def estimate_success_probability(eligibility_status, competition_score,
                                 personal_fit_score) -> dict:
    """Estimate realistic probability of actually winning this.

    Combines eligibility, competition and personal fit multiplicatively. Any
    input that is unknown lowers *confidence* — it never raises the estimate.
    Returns ``{"probability", "confidence", "basis"}``.
    """
    level = normalize_eligibility(eligibility_status)
    p_elig = _ELIGIBILITY_PROBABILITY.get(level, 0.30)

    comp = _clamp(competition_score)
    fit = _clamp(personal_fit_score)

    unknowns = []
    if comp is None:
        # Unknown competition is NOT assumed favourable — assume crowded.
        comp = 2.0
        unknowns.append("competition")
    if fit is None:
        fit = 5.0
        unknowns.append("personal_fit")
    if level == "UNCERTAIN":
        unknowns.append("eligibility")

    p_comp = comp / 10.0
    # Fit dampens the odds but never zeroes them.
    p_fit = 0.5 + 0.5 * (fit / 10.0)

    probability = max(0.0, min(1.0, p_elig * p_comp * p_fit))
    confidence = "high" if not unknowns else ("low" if len(unknowns) > 1 else "medium")
    return {
        "probability": round(probability, 4),
        "confidence": confidence,
        "unknowns": unknowns,
        "basis": {"eligibility": level, "competition_score": comp,
                  "personal_fit_score": fit},
    }


def compute_expected_value(financial_value_score, probability_info,
                           reward_usd=None) -> dict:
    """Risk-adjusted expected value on a 0-10 scale.

    ``expected_value_score = financial_value_score x normalized_probability``.
    Probability is normalized against _EXCELLENT_ODDS so realistic competitive
    odds still produce usable scores.

    ``estimated_value_usd`` is an ESTIMATE for context only — never a promised
    or guaranteed return, and omitted entirely when the reward is unknown.
    """
    fin = _clamp(financial_value_score)
    p = probability_info.get("probability", 0.0)
    p_norm = min(1.0, p / _EXCELLENT_ODDS) if _EXCELLENT_ODDS else 0.0

    if fin is None:
        # Unknown reward size: cannot compute value. Stay uncertain, do not
        # assume the reward (or the odds) are good.
        return {
            "expected_value_score": None,
            "estimated_value_usd": None,
            "probability": p,
            "probability_confidence": probability_info.get("confidence"),
            "is_estimate": True,
            "note": "reward size unknown — expected value not computable",
        }

    ev_score = round(fin * p_norm, 2)
    est_usd = None
    if reward_usd not in (None, "", 0):
        try:
            est_usd = round(float(reward_usd) * p, 2)
        except (TypeError, ValueError):
            est_usd = None

    return {
        "expected_value_score": ev_score,
        "estimated_value_usd": est_usd,
        "probability": p,
        "probability_confidence": probability_info.get("confidence"),
        "is_estimate": True,
        "note": ("estimated expected value, not a guaranteed return; "
                 f"probability confidence: {probability_info.get('confidence')}"),
    }


# A near-worthless expected value caps the final score no matter how
# attractive the other dimensions look. Without this, an unwinnable
# opportunity could still score mid-range on fit/credibility/effort alone.
_EV_CAP_ALLOWANCE = 3.0

# Ceiling for anything we cannot actually vouch for — unverified eligibility,
# or a reward whose size we could not determine. Deliberately below the
# default MIN_SCORE (7) so uncertainty can never become a recommendation.
_UNVERIFIED_CAP = 6.5

# Uncertainty must cost something, otherwise withholding information scores
# better than reporting it honestly.
_CONFIDENCE_MULTIPLIER = {"high": 1.0, "medium": 0.85, "low": 0.7}


def compute_final_score(sub_scores: dict, expected_value: dict,
                        eligibility_level=None, credibility_level=None) -> float:
    """Weighted 0-10 final score built on expected value plus the independent
    dimensions. Weights that have no usable input are dropped and the rest
    re-normalized, so a missing sub-score never silently counts as zero.

    Expected value also acts as a ceiling: something the candidate cannot
    realistically win must not ride a high fit/credibility score into the
    notification threshold.
    """
    ev = expected_value.get("expected_value_score")
    total, weight_used = 0.0, 0.0

    if ev is not None:
        total += FINAL_WEIGHTS["expected_value"] * float(ev)
        weight_used += FINAL_WEIGHTS["expected_value"]

    for field, weight in FINAL_WEIGHTS.items():
        if field == "expected_value":
            continue
        val = _clamp(sub_scores.get(field))
        if val is not None:
            total += weight * val
            weight_used += weight

    if weight_used == 0:
        return 0.0
    weighted = total / weight_used

    # Zero chance of success ⇒ zero practical value, whatever else is true.
    if expected_value.get("probability") == 0:
        return 0.0

    if ev is not None:
        # Cap by expected value: a lottery ticket cannot ride high fit and
        # credibility scores into the notification threshold.
        weighted = min(weighted, _EV_CAP_ALLOWANCE + float(ev))
    else:
        # Reward size unknown ⇒ we cannot say it is worth applying for. Without
        # this, "value unknown" would outrank a known-but-modest reward.
        weighted = min(weighted, _UNVERIFIED_CAP)

    # Penalise estimates built on missing inputs, so withholding data is never
    # more attractive than reporting it honestly.
    weighted *= _CONFIDENCE_MULTIPLIER.get(
        expected_value.get("probability_confidence"), 0.7)

    # Unverified eligibility is never a recommendation to apply.
    if eligibility_level is not None and not meets_threshold(
            eligibility_level, PROBABLY_ELIGIBLE):
        weighted = min(weighted, _UNVERIFIED_CAP)

    # HIGH_RISK must never reach a normal high-priority notification, and
    # anything merely unverified cannot be recommended either.
    if credibility_level is not None:
        cred = normalize_credibility(credibility_level)
        if cred == "HIGH_RISK":
            return 0.0
        if not is_notifiable(cred):
            weighted = min(weighted, _UNVERIFIED_CAP)

    return round(max(0.0, weighted), 1)


def _coerce_sub_scores(raw):
    """Model-supplied ``sub_scores`` → a dict, or ``{}`` when malformed.

    The schema asks for an object, but a model occasionally answers with a
    string ("see reasoning above"), a list, or a bare number. Truthy non-dict
    values used to reach ``.get()`` and raise AttributeError, killing the whole
    candidate at the last stage of the pipeline.

    Malformed output is not a judgement, so it is discarded and the reply is
    treated exactly like one carrying no sub-scores at all: the deterministic
    sub-scores stand on their own. Falsy values (None, "", [], {}) were already
    handled this way and behave identically. Nothing is repaired with another
    model call, and no score is invented.
    """
    if isinstance(raw, dict):
        return raw
    if raw:  # truthy but the wrong shape — worth saying out loud
        print(f"⚠️  scoring model returned {type(raw).__name__} for "
              f"'sub_scores' (expected an object) — ignoring the malformed "
              f"sub-scores and scoring deterministically.")
    return {}


def build_scoring(parsed: dict, prior: dict) -> dict:
    """Derive sub-scores, probability, expected value and final score.

    Model-judged sub-scores are used when present; the rest are derived
    deterministically from the prior analysis (eligibility ladder, legitimacy
    verdict, complexity, deadline) so scoring degrades gracefully rather than
    inventing optimistic numbers.

    ``model_sub_scores_used`` in the result reports whether the model actually
    supplied usable sub-scores, so callers do not have to re-inspect (and
    re-trust) the raw reply.
    """
    model_subs = _coerce_sub_scores(parsed.get("sub_scores"))
    subs = {}

    # Deterministic where we already know the answer from earlier stages.
    elig_status = prior.get("eligibility_status", "UNCERTAIN")
    subs["eligibility_score"] = _ELIGIBILITY_SCORE.get(
        normalize_eligibility(elig_status), 3.0)

    # Credibility comes from the graded status (which already folds in the
    # source tier), falling back to the legacy verdict for older records.
    legit = prior.get("legitimacy") or {}
    cred_status = normalize_credibility(
        legit.get("credibility_status") or legit.get("verdict"))
    subs["credibility_score"] = _CREDIBILITY_SCORE.get(cred_status, 4.0)

    complexity = prior.get("complexity") or {}
    hours = complexity.get("estimated_hours")
    if isinstance(hours, (int, float)) and hours > 0:
        # 5h → ~9, 20h → ~6, 60h → ~2
        subs["effort_score"] = _clamp(10.0 - (math.log10(max(hours, 1)) * 5.0))
    deadline = prior.get("deadline") or {}
    days = deadline.get("days_left")
    if isinstance(days, (int, float)):
        # <7 days → rushed, 60+ days → comfortable
        subs["urgency_score"] = _clamp(days / 6.0)

    # Financial value: dollar figure preferred, funding label as fallback.
    reward_usd = parsed.get("estimated_reward_usd")
    fin = reward_score(reward_usd)
    if fin is None:
        fin = _FUNDING_VALUE_SCORE.get(parsed.get("funding"))
    if fin is not None:
        subs["financial_value_score"] = fin

    # Model judgement wins only for the fields that genuinely need judgement.
    # eligibility_score and credibility_score stay authoritative from the
    # earlier checker stages — the scoring model must not talk itself into a
    # better eligibility or credibility rating than those already established.
    for field in MODEL_JUDGED_FIELDS:
        val = _clamp(model_subs.get(field))
        if val is not None:
            subs[field] = val

    probability = estimate_success_probability(
        elig_status, subs.get("competition_score"), subs.get("personal_fit_score"))
    expected_value = compute_expected_value(
        subs.get("financial_value_score"), probability, reward_usd)
    final_score = compute_final_score(subs, expected_value, elig_status,
                                      cred_status)

    return {
        "sub_scores": {f: (round(subs[f], 2) if isinstance(subs.get(f), (int, float))
                           else subs.get(f))
                       for f in SUB_SCORE_FIELDS},
        "probability": probability,
        "expected_value": expected_value,
        "final_score": final_score,
        "reward_usd": reward_usd,
        "model_sub_scores_used": bool(model_subs),
    }


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
        "Judge each sub-score 0-10. For competition_score, 10 means almost no "
        "competition and 0 means a global contest with thousands of entrants. "
        "For effort/urgency/accessibility, 10 is the easiest case. If you "
        "cannot judge one, return null — do NOT guess a favourable number.\n"
        "Reply ONLY with JSON:\n"
        '{"overall_score": 0-10, '
        '"breakdown": {"fit": 0-3, "funding": 0-3, "winnability": 0-2, '
        '"effort_vs_reward": 0-1, "time": 0-1}, '
        '"sub_scores": {"eligibility_score": 0-10, "personal_fit_score": 0-10, '
        '"credibility_score": 0-10, "financial_value_score": 0-10, '
        '"effort_score": 0-10, "competition_score": 0-10, '
        '"urgency_score": 0-10, "accessibility_score": 0-10}, '
        '"estimated_reward_usd": number or null, '
        '"funding": "fully_funded|partial|none|unknown", '
        '"reasoning": "3-4 sentences, specific to this candidate", '
        '"recommendation": "apply|consider|skip"}\n\n'
        f"OPPORTUNITY TEXT:\n{_trim(data.get('raw_text', ''))}"
    )
    res = call_model("final_scoring", prompt, system=system, max_tokens=900)
    # A non-object reply yields {}, so overall_score is absent, score falls to
    # 0.0 and build_scoring() scores deterministically — nothing is invented.
    parsed = as_object(extract_json(res["content"]), "final_scoring")

    score = parsed.get("overall_score")
    try:
        score = round(float(score), 1)
    except (TypeError, ValueError):
        score = 0.0

    scoring = build_scoring(parsed, prior)

    # Expected-value final score drives the MIN_SCORE gate whenever the model
    # supplied sub-scores. A legacy reply with no sub_scores keeps the model's
    # own overall_score, so older callers/stubs behave exactly as before.
    # Read from the built scoring, not the raw reply: a malformed sub_scores
    # value is truthy but yielded no sub-scores, and must fall back to the
    # model's own overall_score exactly as a missing one does.
    has_model_subs = scoring["model_sub_scores_used"]
    overall = scoring["final_score"] if has_model_subs else score

    return {
        "overall_score": overall,
        "final_score": scoring["final_score"],
        "model_score": score,
        "sub_scores": scoring["sub_scores"],
        "probability": scoring["probability"],
        "expected_value": scoring["expected_value"],
        "breakdown": parsed.get("breakdown", {}),
        "funding": parsed.get("funding", "unknown"),
        "reasoning": parsed.get("reasoning", "No reasoning returned."),
        "recommendation": parsed.get("recommendation", "consider"),
        "model_used": res["model_used"],
        "cost_usd": res["cost_usd"],
    }
