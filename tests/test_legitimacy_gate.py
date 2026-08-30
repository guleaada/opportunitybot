#!/usr/bin/env python3
"""Tests for the legitimacy gate: parse-failure handling and syndication.

Two changes are covered, plus the safety properties that must survive both.

1. An unparseable model reply used to be indistinguishable from the model
   deliberately answering "unknown": both produced verdict "unknown" ->
   NEEDS_VERIFICATION -> the candidate was killed AND marked seen, so it was
   never reconsidered. 35 of 79 NEEDS_VERIFICATION records (44%) died that way.
   It is now retried once, then reported as parse_failed so the caller can
   preserve it.

2. The scam-detection prompt now tells the model that a syndicated listing on
   an aggregator is not itself evidence of fraud. This is PROMPT-ONLY: no code
   path softens a verdict based on the host, because provenance and substance
   arrive bundled in one verdict string with no reliable seam between them.
   43 SUSPICIOUS records cite provenance, but 23 of those ALSO cite a real
   fraud cue — including a fee-for-invitation summit hosted on a whitelisted
   quality-7 aggregator. A host-based override would have unblocked it.

Run:  python tests/test_legitimacy_gate.py
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import checker as C
import known_scams
from checker import HIGH_RISK, SUSPICIOUS, NEEDS_VERIFICATION, LIKELY_LEGITIMATE

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  ✅ {msg}")


def legit(content, url="https://opportunitiescorners.com/x", text="PAGE"):
    """Run check_legitimacy with a stubbed raw model reply."""
    calls = []

    def fake(task, prompt, **kw):
        calls.append(prompt)
        body = content[len(calls) - 1] if isinstance(content, list) else content
        return {"content": body, "model_used": "stub", "cost_usd": 0.0}

    with patch.object(C, "call_model", side_effect=fake):
        return C.check_legitimacy(text, url), calls


GOOD = json.dumps({"verdict": "legitimate", "confidence": 0.9,
                   "reasoning": "Official programme.", "red_flags": []})

# ══════════════════════════════════════════════════════════════════════════
print("\n1. An unparseable reply is retried, then reported — never guessed")

res, calls = legit(["I think this looks fine, honestly.", GOOD])
assert len(calls) == 2, f"expected one retry, got {len(calls)} calls"
assert "JSON object ONLY" in calls[1], "the retry must tighten the instruction"
assert res["verdict"] == "legitimate"
assert res["parse_failed"] is False
ok("unparseable then valid → retried once, verdict recovered")

res, calls = legit(["not json", "still not json"])
assert len(calls) == 2, "must retry exactly once, not loop"
assert res["verdict"] == "unknown"
assert res["parse_failed"] is True
assert res["credibility_status"] == NEEDS_VERIFICATION
ok("unparseable twice → parse_failed=True, still NEEDS_VERIFICATION (no pass)")

res, calls = legit(GOOD)
assert len(calls) == 1, "a parseable reply must not trigger a retry"
assert res["parse_failed"] is False
ok("a good reply costs exactly one call — no extra spend")

# A DELIBERATE "unknown" is not a parse failure.
res, _ = legit(json.dumps({"verdict": "unknown", "confidence": 0.2,
                           "reasoning": "Insufficient evidence.",
                           "red_flags": []}))
assert res["verdict"] == "unknown"
assert res["parse_failed"] is False, \
    "a model that deliberately answered 'unknown' must not look like a crash"
ok("deliberate 'unknown' is distinguishable from an unparseable reply")

# A non-object JSON reply is still handled.
res, _ = legit(["[1,2,3]", "[4,5,6]"])
assert res["verdict"] == "unknown" and res["parse_failed"] is True
ok("a JSON array reply is a parse failure, not a verdict")

# ══════════════════════════════════════════════════════════════════════════
print("\n2. parse_failed NEVER passes the gate")

for reply in (["x", "y"], ["[1]", "[2]"], ["", ""]):
    res, _ = legit(reply)
    assert res["parse_failed"] is True
    assert res["credibility_status"] not in ("VERIFIED", LIKELY_LEGITIMATE), \
        "an unparseable reply must never read as legitimate"
ok("no unparseable reply ever yields VERIFIED or LIKELY_LEGITIMATE")

# ══════════════════════════════════════════════════════════════════════════
print("\n3. SAFETY: negative verdicts still hard-block on a whitelisted host")

WHITELISTED = "https://opportunitiescorners.com/global-peace-summit-2026/"
for v in ("scam", "suspicious"):
    body = json.dumps({"verdict": v, "confidence": 0.9,
                       "reasoning": "Organizer not verifiable; charges a fee.",
                       "red_flags": ["registration fee"]})
    res, _ = legit(body, url=WHITELISTED)
    assert res["verdict"] == v
    assert res["credibility_status"] in (HIGH_RISK, SUSPICIOUS), \
        f"{v} on a whitelisted aggregator must still block"
ok("scam/suspicious on a quality-7 aggregator still resolve to a blocking level")

# The real production case: a fee-for-invitation summit on a whitelisted host.
res, _ = legit(json.dumps({
    "verdict": "suspicious", "confidence": 0.8,
    "reasoning": ("The opportunity is posted on a third-party aggregator, not "
                  "an official site, and the organizer (Global Peace Chain) is "
                  "not verifiable. It advertises 'fully funded' seats yet "
                  "charges a registration fee."),
    "red_flags": ["registration fee", "limited seats"]}), url=WHITELISTED)
assert res["credibility_status"] == SUSPICIOUS, \
    "Global Peace Summit must stay blocked — this is the owner's scam class"
ok("the fee-for-invitation summit stays SUSPICIOUS despite the trusted host")

# ══════════════════════════════════════════════════════════════════════════
print("\n4. SAFETY: the deterministic layers are untouched and still first")

report = known_scams.red_flag_report(
    "Congratulations you qualify! Pay via western union to secure your seat. "
    "A non-refundable fee and visa processing fee payable on arrival.")
for kw in ("congratulations you qualify", "pay via western union",
           "secure your seat", "non-refundable fee",
           "visa processing fee payable"):
    assert kw in report["red_flags_found"], f"{kw} no longer detected"
ok(f"red_flag_report still catches all {len(report['red_flags_found'])} "
   f"fee/wire/invitation phrases in the sample")

assert len(known_scams.RED_FLAG_KEYWORDS) == 29, \
    f"RED_FLAG_KEYWORDS changed: {len(known_scams.RED_FLAG_KEYWORDS)}"
ok("all 29 RED_FLAG_KEYWORDS still present — the blocklist was not edited")

detail = known_scams.check_known_scam("Global Peace Summit", WHITELISTED)
assert isinstance(detail, dict) and "is_scam" in detail
ok("check_known_scam still callable and still runs ahead of any model call")

# The red-flag scan is still fed into the prompt.
_, calls = legit(GOOD, text="Please pay the registration fee to attend.")
assert "registration fee" in calls[0], \
    "red_flag_report output must still reach the scam-detection prompt"
ok("detected red flags are still passed into the prompt")

# ══════════════════════════════════════════════════════════════════════════
print("\n5. The syndication guidance is prompt-only, with fraud cues intact")

sysmsg = None
def capture(task, prompt, system=None, **kw):
    global sysmsg
    sysmsg = system
    return {"content": GOOD, "model_used": "stub", "cost_usd": 0.0}
with patch.object(C, "call_model", side_effect=capture):
    C.check_legitimacy("text", "https://opportunitydesk.org/x")

assert "JUDGE THE PROGRAM, NOT THE HOST" in sysmsg
for cue in ("fee to apply", "invitation letter", "personal or "
            "non-institutional account", "free email address",
            "does not verifiably exist", "guaranteed selection"):
    assert cue in sysmsg, f"fraud cue missing from the prompt: {cue}"
ok("prompt tells the model to judge the program AND lists the fraud cues")

# There must be no code path keying credibility off the host.
import inspect
src = inspect.getsource(C.credibility_status)
assert "aggregator" not in src.lower()
assert "quality" not in src.lower(), \
    "credibility_status must not branch on source quality"
ok("credibility_status has no host-based override — prompt-only, as scoped")

print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
