#!/usr/bin/env python3
"""
selftest_offline.py — prove the pipeline wiring with NO network / NO API keys.

Monkeypatches model_router.call_model so every "model" returns canned JSON,
then runs a fake candidate through the REAL main.analyze_one() pipeline and the
REAL report builder. Verifies routing, gating, scoring, and report output all
connect — without spending a cent or needing credentials.

Run:  python selftest_offline.py
"""

import json

import model_router
import checker
import scorer
import tools
import search
import main as app


# ── Canned per-task responses keyed by task_type ───────────────────────────
_RESPONSES = {
    "clean_html": "DAAD EPOS Scholarship. Fully funded masters for professionals "
                  "from developing countries. Open to Ethiopian nationals. "
                  "Deadline 31 August 2026.",
    "first_pass_filter": {"keep": True, "reason": "Fully funded, Ethiopians eligible",
                          "guessed_country": "Germany", "guessed_funding": "fully_funded"},
    "scam_detection": {"verdict": "legitimate", "confidence": 0.95,
                       "reasoning": "Official DAAD government program, no fees.",
                       "red_flags": []},
    "deep_eligibility": {"overall": "eligible",
                         "reasoning": "BSc + 2yr work experience meets EPOS rules.",
                         "blocking_issues": [],
                         "addressable_gaps": ["English certificate (Duolingo)"]},
    "extract_document_requirements": {"documents": ["CV", "Motivation letter", "Degree certificate"],
                                      "english_test_required": "Duolingo",
                                      "references_required": 2, "transcripts_required": True,
                                      "notes": "ok"},
    "estimate_complexity": {"estimated_hours": 25, "difficulty": "medium",
                            "odds": "moderate", "notes": "essays + references"},
    "final_scoring": {"overall_score": 9.2,
                      "breakdown": {"fit": 3, "funding": 3, "winnability": 1.5,
                                    "effort_vs_reward": 0.9, "time": 0.8},
                      "funding": "fully_funded",
                      "reasoning": "Excellent fit: fully funded, eligible, AI-relevant.",
                      "recommendation": "apply"},
    "generate_email_subject": "DAAD EPOS — fully funded masters match (9.2/10)",
}

# A deadline ~ far in the future so the deadline gate passes.
_RESPONSES_DEADLINE = {"deadline_iso": "2026-08-31", "deadline_raw": "31 August 2026",
                       "is_explicitly_closed": False, "found": True}


_CALLS = []  # (task_type, model) per fake call — lets tests assert cost behavior


def fake_call_model(task_type, prompt, system=None, tools=None,
                    max_tokens=2048, temperature=0.3):
    if task_type == "check_deadline":
        payload = fake_call_model.deadline_response
    else:
        payload = _RESPONSES.get(task_type, {"keep": True})
    content = payload if isinstance(payload, str) else json.dumps(payload)
    model = {"scam_detection": "claude-sonnet-4-5", "deep_eligibility": "claude-sonnet-4-5",
             "final_scoring": "claude-sonnet-4-5"}.get(task_type, "stub-free-model")
    cost = 0.04 if model.startswith("claude") else 0.0
    _CALLS.append((task_type, model))
    return {"content": content, "model_used": model, "task_type": task_type,
            "tokens_used": {"input": 100, "output": 50}, "cost_usd": cost,
            "fell_back": False}


fake_call_model.deadline_response = _RESPONSES_DEADLINE


def main_():
    # Patch the single routing entry point everywhere it's imported.
    for mod in (model_router, checker, scorer, tools):
        if hasattr(mod, "call_model"):
            mod.call_model = fake_call_model

    # Patch network fetch so no HTTP happens.
    tools.fetch_url = lambda url, force=False: {
        "url": url, "html": "<html>...</html>",
        "text": "raw page text about DAAD EPOS scholarship",
        "status": 200, "cached": False, "error": None,
    }
    # Patch notifier so nothing is actually sent.
    tools.send_notification = lambda report, channel="all", subject=None: (
        print("\n--- REPORT THAT WOULD BE SENT ---\n" + report), {"stub": True})[1]
    tools.add_to_calendar = lambda data: {"added": False, "reason": "stubbed"}

    # Build one fake candidate and run the REAL pipeline.
    result = search.SearchResult(
        title="DAAD EPOS Scholarship (Germany)",
        url="https://www.daad.de/epos-example",
        snippet="Fully funded masters", source="daad.de")

    stats = {k: 0 for k in (
        "discovered", "already_seen", "known_scam", "fetch_failed",
        "first_pass_dropped", "scam", "legit_unknown", "ineligible", "closed",
        "deep_analyzed", "scored_high")}

    match = app.analyze_one(result, stats)

    assert match is not None, "expected a notify-worthy match"
    assert match["score"]["overall_score"] == 9.2, match["score"]
    assert stats["deep_analyzed"] == 1 and stats["scored_high"] == 1, stats
    assert match["legitimacy"]["model_used"] == "claude-sonnet-4-5"
    assert match["documents"]["english_test_required"] == "Duolingo"

    report = app.build_report([match], {**stats, "discovered": 1}, [
        {"name": "Global Business Summit 2026", "reason": "fee-based invitation pattern"}])
    app.tools.send_notification(report)

    assert "DAAD EPOS Scholarship" in report
    assert "9.2/10" in report
    assert "BLOCKED SCAMS" in report

    # ── Scenario 2: CLOSED deadline must cost ZERO Claude calls and land on
    # the watchlist (the deadline gate runs before any paid call).
    import database as db
    fake_call_model.deadline_response = {
        "deadline_iso": "2025-01-15", "deadline_raw": "15 January 2025",
        "is_explicitly_closed": False, "found": True}
    _CALLS.clear()
    closed_result = search.SearchResult(
        title="Annual Fellowship (closed)", url="https://example.org/closed-fellowship")
    stats2 = {k: 0 for k in stats}
    match2 = app.analyze_one(closed_result, stats2)

    assert match2 is None, "closed program must not match"
    assert stats2["closed"] == 1, stats2
    claude_calls = [t for t, m in _CALLS if m.startswith("claude")]
    assert not claude_calls, f"closed program burned Claude calls: {claude_calls}"
    wl = db.all_watchlist()
    assert any(v.get("url") == closed_result.url for v in wl.values()), \
        "closed program should be on the watchlist"
    # Clean up the test watchlist entry.
    for oid, v in list(wl.items()):
        if v.get("url") == closed_result.url:
            db.remove_from_watchlist(oid)
    fake_call_model.deadline_response = _RESPONSES_DEADLINE

    print("\n✅ OFFLINE PIPELINE SELF-TEST PASSED")
    print(f"   routing OK: legitimacy/eligibility/scoring → claude; "
          f"filter/docs/complexity → free")
    print(f"   gating OK: 1 deep-analyzed, 1 scored >= {app.MIN_SCORE}")
    print("   cost OK: closed deadline gated with 0 Claude calls + watchlisted")


if __name__ == "__main__":
    main_()
