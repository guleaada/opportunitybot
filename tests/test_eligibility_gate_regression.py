#!/usr/bin/env python3
"""Regression tests for the eligibility gate that stalled the scan.

Between 2026-08-15 (commit 2c33529, "Add graded eligibility") and 2026-08-28,
177 candidates were graded and exactly 2 cleared the gate — 1.1%, against
24.3% in the two months before the ladder landed. No candidate reached the
scorer, so the bot produced zero matches for two weeks while discovery volume
from the two productive feeds was unchanged (4.9 vs 4.8 records/day).

Cause: Rule 2 in check_eligibility collapsed ANY positive grade to UNCERTAIN
whenever missing_requirements was non-empty. Every real program page leaves
something unstated, so the rule fired on nearly every candidate and
PROBABLY_ELIGIBLE — the level main.py gates on — became unreachable.

These tests pin the new behaviour: an unverified detail costs at most one step
on the ladder and never vetoes the verdict, while every path that could turn
missing information INTO eligibility stays closed.

Run:  python tests/test_eligibility_gate_regression.py
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import checker as C
from checker import (
    CONFIRMED_ELIGIBLE, PROBABLY_ELIGIBLE, UNCERTAIN,
    PROBABLY_INELIGIBLE, CONFIRMED_INELIGIBLE, meets_threshold,
)

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  ✅ {msg}")


def grade(**payload):
    """Run check_eligibility with a stubbed model reply."""
    reply = {"content": json.dumps(payload), "model_used": "stub",
             "cost_usd": 0.0}
    with patch.object(C, "call_model", return_value=reply):
        return C.check_eligibility("PROGRAM TEXT", {"name": "test"})


# ══════════════════════════════════════════════════════════════════════════
# 1. The exact production shape: positive grade + unstated detail
# ══════════════════════════════════════════════════════════════════════════
print("\n1. A positive grade survives an unverified detail")

res = grade(eligibility_status="PROBABLY_ELIGIBLE",
            reasoning="Nothing on the page excludes the candidate.",
            missing_requirements=["English test score not stated"],
            citizenship_mismatch=False)
assert res["eligibility_status"] == PROBABLY_ELIGIBLE, res["eligibility_status"]
assert meets_threshold(res["eligibility_status"], PROBABLY_ELIGIBLE)
ok("PROBABLY_ELIGIBLE + missing_requirements stands (was demoted to UNCERTAIN)")

res = grade(eligibility_status="CONFIRMED_ELIGIBLE",
            reasoning="All stated requirements met.",
            missing_requirements=["reference count not stated"],
            citizenship_mismatch=False)
assert res["eligibility_status"] == PROBABLY_ELIGIBLE, res["eligibility_status"]
assert meets_threshold(res["eligibility_status"], PROBABLY_ELIGIBLE)
ok("CONFIRMED_ELIGIBLE + missing → PROBABLY_ELIGIBLE (one step, still passes)")

res = grade(eligibility_status="CONFIRMED_ELIGIBLE",
            reasoning="Everything verified on the page.",
            missing_requirements=[], citizenship_mismatch=False)
assert res["eligibility_status"] == CONFIRMED_ELIGIBLE
ok("CONFIRMED_ELIGIBLE with nothing missing is left alone")

# The production casualty, replayed verbatim.
res = grade(eligibility_status="PROBABLY_ELIGIBLE",
            reasoning=("The program does not explicitly state nationality "
                       "restrictions, student status requirements, age limits "
                       "or degree level requirements. The candidate is an "
                       "Ethiopian national (not excluded), is a working "
                       "professional."),
            missing_requirements=["nationality restrictions", "age limits",
                                  "degree level"],
            citizenship_mismatch=False)
assert meets_threshold(res["eligibility_status"], PROBABLY_ELIGIBLE), res
ok("GEAF Ambassador Programme (real Aug-2026 rejection) now reaches the scorer")

# ══════════════════════════════════════════════════════════════════════════
# 2. Missing information still cannot MANUFACTURE eligibility
# ══════════════════════════════════════════════════════════════════════════
print("\n2. The honesty rule holds — a level is only ever lowered")

for level in (UNCERTAIN, PROBABLY_INELIGIBLE, CONFIRMED_INELIGIBLE):
    res = grade(eligibility_status=level, reasoning="r",
                missing_requirements=["lots", "of", "unknowns"],
                citizenship_mismatch=False)
    assert res["eligibility_status"] == level, (level, res["eligibility_status"])
    assert not meets_threshold(res["eligibility_status"], PROBABLY_ELIGIBLE)
ok("UNCERTAIN / PROBABLY_INELIGIBLE / CONFIRMED_INELIGIBLE are never upgraded")

res = grade(reasoning="model omitted the grade entirely",
            missing_requirements=[], citizenship_mismatch=False)
assert res["eligibility_status"] == UNCERTAIN
assert not meets_threshold(res["eligibility_status"], PROBABLY_ELIGIBLE)
ok("a missing eligibility_status still defaults to UNCERTAIN and fails the gate")

with patch.object(C, "call_model",
                  return_value={"content": "[1, 2, 3]", "model_used": "stub",
                                "cost_usd": 0.0}):
    res = C.check_eligibility("text", {})
assert res["eligibility_status"] == UNCERTAIN
ok("a non-object JSON reply still fails closed at UNCERTAIN")

# ══════════════════════════════════════════════════════════════════════════
# 3. Rule 1 (citizenship) is untouched and still outranks everything
# ══════════════════════════════════════════════════════════════════════════
print("\n3. A hard citizenship mismatch still disqualifies outright")

for level in (CONFIRMED_ELIGIBLE, PROBABLY_ELIGIBLE, UNCERTAIN):
    res = grade(eligibility_status=level, reasoning="r",
                missing_requirements=[], citizenship_mismatch=True)
    assert res["eligibility_status"] == CONFIRMED_INELIGIBLE, (level, res)
ok("citizenship_mismatch → CONFIRMED_INELIGIBLE from any starting level")

res = grade(eligibility_status="CONFIRMED_ELIGIBLE", reasoning="r",
            missing_requirements=["x"], citizenship_mismatch=True)
assert res["eligibility_status"] == CONFIRMED_INELIGIBLE
ok("Rule 1 wins over Rule 2 when both apply")

# ══════════════════════════════════════════════════════════════════════════
# 4. The demotion is observable, so the next run can be attributed
# ══════════════════════════════════════════════════════════════════════════
print("\n4. model_eligibility_status records the pre-rule grade")

res = grade(eligibility_status="CONFIRMED_ELIGIBLE", reasoning="r",
            missing_requirements=["x"], citizenship_mismatch=False)
assert res["model_eligibility_status"] == CONFIRMED_ELIGIBLE
assert res["eligibility_status"] == PROBABLY_ELIGIBLE
ok("a demoted verdict keeps the model's original grade for attribution")

res = grade(eligibility_status="UNCERTAIN", reasoning="r",
            missing_requirements=["x"], citizenship_mismatch=False)
assert res["model_eligibility_status"] == res["eligibility_status"] == UNCERTAIN
ok("an UNCERTAIN the model chose itself is distinguishable from a demotion")

res = grade(eligibility_status="PROBABLY_ELIGIBLE", reasoning="r",
            missing_requirements=["x"], citizenship_mismatch=True)
assert res["model_eligibility_status"] == PROBABLY_ELIGIBLE
assert res["eligibility_status"] == CONFIRMED_INELIGIBLE
ok("a Rule 1 override is attributable too")

# ══════════════════════════════════════════════════════════════════════════
# 5. Contract kept: legacy 'overall', list coercion, required keys
# ══════════════════════════════════════════════════════════════════════════
print("\n5. The returned contract is unchanged for existing readers")

res = grade(eligibility_status="PROBABLY_ELIGIBLE", reasoning="r",
            missing_requirements=["x"], citizenship_mismatch=False)
assert res["overall"] == "eligible", res["overall"]
for key in ("eligibility_status", "model_eligibility_status", "overall",
            "reasoning", "blocking_issues", "addressable_gaps",
            "missing_requirements", "citizenship_mismatch", "model_used"):
    assert key in res, f"missing key {key}"
ok("every documented key is present; 'overall' still maps to the ladder")

res = grade(eligibility_status="CONFIRMED_ELIGIBLE", reasoning="r",
            missing_requirements="a bare string, not a list",
            citizenship_mismatch=False)
assert res["missing_requirements"] == ["a bare string, not a list"]
assert res["eligibility_status"] == PROBABLY_ELIGIBLE
ok("a non-list missing_requirements is still coerced, and still demotes")

# ══════════════════════════════════════════════════════════════════════════
# 6. The gate itself is unchanged — only what reaches it
# ══════════════════════════════════════════════════════════════════════════
print("\n6. The >= PROBABLY_ELIGIBLE threshold is untouched")

assert meets_threshold(CONFIRMED_ELIGIBLE, PROBABLY_ELIGIBLE)
assert meets_threshold(PROBABLY_ELIGIBLE, PROBABLY_ELIGIBLE)
assert not meets_threshold(UNCERTAIN, PROBABLY_ELIGIBLE)
assert not meets_threshold(PROBABLY_INELIGIBLE, PROBABLY_ELIGIBLE)
assert not meets_threshold(CONFIRMED_INELIGIBLE, PROBABLY_ELIGIBLE)
ok("UNCERTAIN and below still never reach the scorer")

print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
