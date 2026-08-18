#!/usr/bin/env python3
"""Regression tests: malformed ``sub_scores`` must not crash the scorer.

The scoring model answers with JSON whose ``sub_scores`` should be an object.
When it answered with a string instead, ``model_subs.get(field)`` raised
AttributeError and the candidate died at the very last stage of the pipeline.

Malformed sub-scores are now discarded and the reply is treated exactly like
one with no sub-scores: the deterministic sub-scores (eligibility ladder,
credibility, complexity, deadline) stand on their own, and the MIN_SCORE gate
falls back to the model's own overall_score. Nothing is repaired with an extra
model call and no score is invented.

Run:  python tests/test_scorer_malformed_subscores.py
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scorer

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  ✅ {msg}")


PRIOR = {
    "legitimacy": {"credibility_status": "VERIFIED", "verdict": "legitimate"},
    "eligibility_status": "CONFIRMED_ELIGIBLE",
    "eligibility": {"eligibility_status": "CONFIRMED_ELIGIBLE"},
    "documents": {"documents": ["CV"]},
    "complexity": {"estimated_hours": 20},
    "deadline": {"days_left": 45},
}
GOOD_SUBS = {
    "eligibility_score": 9.0, "personal_fit_score": 9.0,
    "credibility_score": 9.0, "financial_value_score": 9.0,
    "effort_score": 6.0, "competition_score": 3.0,
    "urgency_score": 7.0, "accessibility_score": 8.0,
}


def base(**over):
    p = {"overall_score": 8.4, "estimated_reward_usd": 30000,
         "funding": "fully_funded", "breakdown": {}, "reasoning": "r",
         "recommendation": "apply"}
    p.update(over)
    return p


# ══════════════════════════════════════════════════════════════════════════
# 1. dict → processed normally (the reference behaviour nothing may change)
# ══════════════════════════════════════════════════════════════════════════
print("\n1. sub_scores is a dict → processed normally")
ref = scorer.build_scoring(base(sub_scores=GOOD_SUBS), PRIOR)
assert ref["model_sub_scores_used"] is True
for field in scorer.MODEL_JUDGED_FIELDS:
    assert ref["sub_scores"][field] == GOOD_SUBS[field], (field, ref["sub_scores"])
# eligibility/credibility stay authoritative from the earlier stages.
assert ref["sub_scores"]["eligibility_score"] == \
    scorer._ELIGIBILITY_SCORE["CONFIRMED_ELIGIBLE"]
ok(f"model-judged fields honoured; final_score {ref['final_score']}")

# ══════════════════════════════════════════════════════════════════════════
# 2. Every malformed / absent shape: no crash, deterministic fallback
# ══════════════════════════════════════════════════════════════════════════
print("\n2. Malformed and absent sub_scores never raise")
MISSING = object()
CASES = [
    ("missing",        MISSING),
    ("null",           None),
    ("string",         "see the reasoning above"),
    ("empty string",   ""),
    ("list",           [9, 8, 7]),
    ("empty list",     []),
    ("list of dicts",  [{"personal_fit_score": 9}]),
    ("int",            7),
    ("float",          7.5),
    ("bool",           True),
    ("empty dict",     {}),
    ("nested string",  "{\"personal_fit_score\": 9}"),
]

baseline = None
for label, value in CASES:
    parsed = base() if value is MISSING else base(sub_scores=value)
    try:
        got = scorer.build_scoring(parsed, PRIOR)
    except Exception as e:                       # noqa: BLE001 — that's the bug
        raise AssertionError(f"{label} sub_scores raised {type(e).__name__}: {e}")
    assert got["model_sub_scores_used"] is False, label
    # No model-judged field may be invented out of malformed output.
    for field in ("personal_fit_score", "competition_score",
                  "accessibility_score"):
        assert got["sub_scores"][field] is None, (label, field, got["sub_scores"])
    # Every malformed shape must land on the SAME result as a missing one.
    if baseline is None:
        baseline = got
    else:
        assert got == baseline, (label, got, baseline)
ok(f"{len(CASES)} shapes (dict/missing/null/string/list/int/bool/...) — "
   f"no crash, all identical to 'missing'")

assert baseline["sub_scores"]["eligibility_score"] == \
    scorer._ELIGIBILITY_SCORE["CONFIRMED_ELIGIBLE"]
assert baseline["sub_scores"]["credibility_score"] == \
    scorer._CREDIBILITY_SCORE["VERIFIED"]
assert baseline["sub_scores"]["urgency_score"] is not None
assert baseline["sub_scores"]["effort_score"] is not None
ok("deterministic sub-scores (eligibility/credibility/effort/urgency) still derived")

# ══════════════════════════════════════════════════════════════════════════
# 3. Dicts with junk values still work — _clamp already guards those
# ══════════════════════════════════════════════════════════════════════════
print("\n3. A dict whose values are junk keeps the good ones")
mixed = scorer.build_scoring(
    base(sub_scores={"personal_fit_score": 9.0,
                     "competition_score": "high",
                     "effort_score": None,
                     "accessibility_score": [1],
                     "urgency_score": 99}), PRIOR)
assert mixed["sub_scores"]["personal_fit_score"] == 9.0
assert mixed["sub_scores"]["competition_score"] is None
assert mixed["sub_scores"]["urgency_score"] == 10.0, "99 must clamp to 10"
assert mixed["model_sub_scores_used"] is True
ok("string/None/list values ignored per-field; out-of-range values clamped")

# ══════════════════════════════════════════════════════════════════════════
# 4. Scores are unchanged for well-formed replies (no meaning drift)
# ══════════════════════════════════════════════════════════════════════════
print("\n4. Well-formed scoring is bit-identical to before the fix")
# Captured from the pre-fix implementation on this exact input. Any drift here
# means the fix changed the meaning of a score, which it must not.
assert ref["final_score"] == 7.0, ref["final_score"]
assert ref["probability"]["probability"] == 0.285, ref["probability"]
assert ref["probability"]["confidence"] == "high", ref["probability"]
assert ref["expected_value"]["expected_value_score"] == 5.13, ref["expected_value"]
assert ref["expected_value"]["estimated_value_usd"] == 8550.0, ref["expected_value"]
assert ref["reward_usd"] == 30000
assert ref["sub_scores"]["eligibility_score"] == 10.0
assert ref["sub_scores"]["credibility_score"] == 10.0
ok("known-good case still scores final 7.0 / p 0.285 / EV 5.13 / $8550")

# ══════════════════════════════════════════════════════════════════════════
# 5. End-to-end through score_opportunity — the path that actually crashed
# ══════════════════════════════════════════════════════════════════════════
print("\n5. score_opportunity() survives a malformed reply")


def reply(payload):
    return {"content": json.dumps(payload), "model_used": "stub",
            "task_type": "final_scoring", "tokens_used": {"input": 1, "output": 1},
            "cost_usd": 0.0, "fell_back": False}


DATA = {"raw_text": "Fully funded fellowship.", "url": "https://example.org/f",
        "legitimacy": PRIOR["legitimacy"], "eligibility": PRIOR["eligibility"],
        "documents": PRIOR["documents"], "complexity": PRIOR["complexity"],
        "deadline": PRIOR["deadline"]}

calls = []


def counting(*a, **k):
    calls.append(k.get("task_type") or a[0])
    return counting.response


with patch.object(scorer, "call_model", counting), \
     patch.object(scorer, "profile_summary", return_value="profile"):
    counting.response = reply(base(sub_scores="the scores are in my reasoning"))
    out_bad = scorer.score_opportunity(DATA, {})
    counting.response = reply(base())                       # no sub_scores
    out_missing = scorer.score_opportunity(DATA, {})
    counting.response = reply(base(sub_scores=GOOD_SUBS))
    out_good = scorer.score_opportunity(DATA, {})

assert len(calls) == 3, f"scorer made {len(calls)} model calls for 3 scorings"
ok("no extra model call is made to repair malformed output")

# A malformed reply falls back to the model's own overall_score, exactly as a
# missing one does — the gate meaning does not change.
assert out_bad["overall_score"] == out_missing["overall_score"] == 8.4
assert out_bad["model_score"] == 8.4
assert out_bad["sub_scores"] == out_missing["sub_scores"]
ok("malformed reply → same result as a reply with no sub_scores (8.4)")

# A well-formed reply still switches to the expected-value final score.
assert out_good["overall_score"] == out_good["final_score"] == 7.0
assert out_good["model_score"] == 8.4, "the model's own score is still reported"
assert out_good["overall_score"] != out_good["model_score"]
ok("well-formed reply still gated on the expected-value score (7.0, not 8.4)")

for out in (out_bad, out_missing, out_good):
    assert set(out) >= {"overall_score", "final_score", "model_score",
                        "sub_scores", "probability", "expected_value",
                        "breakdown", "funding", "reasoning",
                        "recommendation", "model_used", "cost_usd"}
ok("return shape unchanged in every case")

print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
