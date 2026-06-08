#!/usr/bin/env python3
"""
main.py — OpportunityBot runner + CLI.

Multi-model AI agent that finds fellowships/scholarships matching the profile
in profile.py, filters scams and closed/ineligible programs, and notifies only
about genuine open opportunities scoring >= MIN_SCORE_TO_NOTIFY.

The pipeline is deliberately ordered to MINIMIZE paid Claude calls: cheap free
models (Groq/Gemini) discard most candidates before any Claude token is spent.

CLI:
  python main.py --scan          Full daily scan
  python main.py --url <URL>     Analyze one specific URL end-to-end
  python main.py --watchlist     Re-check watchlist for newly-opened programs
  python main.py --tracker       Show the application tracker
  python main.py --cost          Show monthly cost breakdown by provider
  python main.py --daemon        Run scheduled daily at DAILY_SCAN_TIME
  python main.py --test          Smoke-test all 3 model providers
"""

import argparse
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()  # load .env before importing modules that read os.getenv at import

import tools
from profile import PROFILE
from source_whitelist import load_whitelist
from cost_tracker import (
    get_daily_summary, get_monthly_summary, get_daily_claude_spend,
    get_monthly_claude_spend,
)
import database as db

try:
    from rich.console import Console
    _console = Console()

    def cprint(*a, **k):
        _console.print(*a, **k)
except ImportError:  # pragma: no cover
    def cprint(*a, **k):
        print(*a)


MIN_SCORE = int(os.getenv("MIN_SCORE_TO_NOTIFY", "7"))
MAX_PER_DAY = int(os.getenv("MAX_OPPORTUNITIES_PER_DAY", "15"))


# ════════════════════════════════════════════════════════════════════════
# Core pipeline
# ════════════════════════════════════════════════════════════════════════
def analyze_one(result, stats: dict):
    """Run the full triage pipeline on a single search result.

    Returns a rich dict if it's a notify-worthy match, else None. Mutates
    ``stats`` counters along the way.
    """
    url = result.url
    title = result.title

    # 4. fetch (no model)
    fetched = tools.fetch_url(url)
    if fetched["error"] or fetched["status"] != 200 or not fetched["text"]:
        db.log_error(f"Fetch failed for {url}: "
                     f"{fetched['error'] or fetched['status']}")
        stats["fetch_failed"] += 1
        return None

    # 5. clean_html (GROQ free) — best effort; falls back to raw text
    text = tools.clean_html(fetched["text"])

    # 6. first_pass_filter (GEMINI free) — drops most candidates
    filt = tools.first_pass_filter(text, PROFILE)
    if not filt["keep"]:
        stats["first_pass_dropped"] += 1
        tools.save_opportunity({"url": url, "title": title,
                                "status": "filtered_out", "reason": filt["reason"]})
        cprint(f"   ✂️  dropped: {filt['reason']}")
        return None

    # 7. check_legitimacy (CLAUDE $) — only on survivors
    legit = tools.check_legitimacy(text, url)
    if legit["verdict"] == "scam":
        stats["scam"] += 1
        tools.save_opportunity({"url": url, "title": title, "status": "scam",
                                "reasoning": legit["reasoning"]})
        cprint(f"   🚫 scam: {legit['reasoning']}")
        return None
    if legit["verdict"] == "unknown":
        stats["legit_unknown"] += 1
        tools.save_opportunity({"url": url, "title": title,
                                "status": "legitimacy_unknown",
                                "reasoning": legit["reasoning"]})
        cprint("   ❓ legitimacy unknown — skipping (honesty rule)")
        return None

    # 8. check_eligibility (CLAUDE $)
    elig = tools.check_eligibility(text, PROFILE)
    if elig["overall"] != "eligible":
        stats["ineligible"] += 1
        tools.save_opportunity({"url": url, "title": title,
                                "status": elig["overall"],
                                "reasoning": elig["reasoning"]})
        cprint(f"   ⛔ {elig['overall']}: {elig['reasoning']}")
        return None

    # 9-10. enrichment (GEMINI free)
    docs = tools.extract_documents(text)
    complexity = tools.estimate_complexity(text)
    deadline = tools.check_deadline(text)

    if deadline["status"] == "closed":
        stats["closed"] += 1
        tools.save_opportunity({"url": url, "title": title, "status": "closed",
                                "deadline": deadline.get("deadline")})
        cprint("   📕 deadline closed — skipping")
        return None

    # 11. score_opportunity (CLAUDE $)
    score = tools.score_opportunity({
        "raw_text": text, "url": url,
        "legitimacy": legit, "eligibility": elig,
        "documents": docs, "complexity": complexity, "deadline": deadline,
    }, PROFILE)
    stats["deep_analyzed"] += 1

    tools.save_opportunity({
        "url": url, "title": title, "status": "analyzed",
        "score": score["overall_score"], "deadline": deadline.get("deadline"),
    })

    if score["overall_score"] >= MIN_SCORE:
        stats["scored_high"] += 1
        return {
            "url": url, "title": title, "score": score,
            "documents": docs, "complexity": complexity,
            "eligibility": elig, "legitimacy": legit, "deadline": deadline,
        }
    cprint(f"   📉 scored {score['overall_score']} (< {MIN_SCORE}) — not notifying")
    return None


def run_scan(max_results_per_source: int = 8):
    """Daily scan pipeline. Returns (final_opportunities, stats)."""
    cprint(f"🚀 [bold]Starting scan[/bold] at {datetime.now().isoformat(timespec='seconds')}"
           if _has_rich() else f"🚀 Starting scan at {datetime.now()}")
    db.log_activity("scan_start", "daily scan started")

    stats = {
        "discovered": 0, "already_seen": 0, "known_scam": 0,
        "fetch_failed": 0, "first_pass_dropped": 0, "scam": 0,
        "legit_unknown": 0, "ineligible": 0, "closed": 0,
        "deep_analyzed": 0, "scored_high": 0,
    }
    blocked_scams = []

    # 1. search across whitelisted sources (no model)
    sources = load_whitelist()
    all_results = []
    seen_urls = set()
    for source in sources:
        try:
            results = tools.web_search(source.search_query, max_results=max_results_per_source)
            for r in results:
                if r.url and r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)
        except Exception as e:
            db.log_error(f"Source {source.name} failed: {e}")
    stats["discovered"] = len(all_results)

    # 2-3. dedupe seen + hard-block known scams (no model)
    candidates = []
    for r in all_results:
        if tools.is_already_seen(r.url):
            stats["already_seen"] += 1
            continue
        detail = tools.known_scam_detail(r.title, r.url)
        if detail["is_scam"]:
            stats["known_scam"] += 1
            blocked_scams.append({"name": r.title or detail["matched"],
                                  "reason": detail["reason"]})
            tools.save_opportunity({"url": r.url, "title": r.title,
                                    "status": "known_scam",
                                    "reasoning": detail["reason"]})
            continue
        candidates.append(r)

    cprint(f"📊 Found {stats['discovered']} total, {len(candidates)} new candidates "
           f"({stats['already_seen']} seen, {stats['known_scam']} blocked)")

    # 4-11. deep pipeline (budget-bounded by MAX_PER_DAY)
    final_opportunities = []
    for r in candidates[:MAX_PER_DAY]:
        cprint(f"🔎 {r.title[:70]}")
        try:
            match = analyze_one(r, stats)
            if match:
                final_opportunities.append(match)
        except Exception as e:
            db.log_error(f"Failed to analyze {r.url}: {e}")
            continue

    # 12. report + notify
    final_opportunities.sort(key=lambda m: m["score"]["overall_score"], reverse=True)
    report = build_report(final_opportunities, stats, blocked_scams)
    top_name = final_opportunities[0]["title"] if final_opportunities else ""
    subject = tools.generate_email_subject(len(final_opportunities), top_name)
    tools.send_notification(report, channel="all", subject=subject)

    # calendar
    for opp in final_opportunities:
        try:
            tools.add_to_calendar({"title": opp["title"], "url": opp["url"],
                                   "deadline": opp["deadline"].get("deadline")})
        except Exception as e:
            db.log_error(f"Calendar add failed for {opp['url']}: {e}")

    db.log_activity("scan_complete", "daily scan finished",
                    matches=len(final_opportunities), stats=stats)

    daily = get_daily_summary()
    cprint(f"💰 Today's spend: Claude=${daily['claude']['cost']:.3f} "
           f"({daily['claude']['calls']} calls), Gemini=FREE "
           f"({daily['gemini']['calls']}), Groq=FREE ({daily['groq']['calls']})")
    return final_opportunities, stats


def _has_rich():
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


# ════════════════════════════════════════════════════════════════════════
# Report builder
# ════════════════════════════════════════════════════════════════════════
def build_report(opportunities, stats, blocked_scams):
    today = datetime.now(timezone.utc).strftime("%B %d, %Y")
    daily = get_daily_summary()
    monthly_claude = get_monthly_claude_spend()
    monthly_cap = float(os.getenv("MONTHLY_CLAUDE_BUDGET_USD", "10.00"))
    pct = (monthly_claude / monthly_cap * 100) if monthly_cap else 0

    free_calls = daily["gemini"]["calls"] + daily["groq"]["calls"]
    lines = [
        f"🎯 OpportunityBot Daily Report — {today}",
        "",
        "📊 SCAN STATS",
        f"   Discovered:            {stats['discovered']}",
        f"   Skipped already-seen:  {stats['already_seen']}",
        f"   Hard-blocked scams:    {stats['known_scam']}",
        f"   First-pass filtered:   {stats['first_pass_dropped']}",
        f"   Flagged scam (Claude): {stats['scam']}",
        f"   Ineligible:            {stats['ineligible']}",
        f"   Closed deadline:       {stats['closed']}",
        f"   Deep-analyzed (Claude):{stats['deep_analyzed']}",
        f"   Scoring >= {MIN_SCORE}:         {stats['scored_high']}",
        "",
        "💰 COST TODAY",
        f"   Claude: ${daily['claude']['cost']:.3f} ({daily['claude']['calls']} calls)",
        f"   Gemini: $0.00 ({daily['gemini']['calls']} free calls)",
        f"   Groq:   $0.00 ({daily['groq']['calls']} free calls)",
        f"   Total:  ${daily['total_cost']:.3f}",
        "",
        "📈 MONTH-TO-DATE",
        f"   Claude: ${monthly_claude:.2f} / ${monthly_cap:.2f} budget ({pct:.0f}% used)",
        "",
        "═" * 51,
        f"✅ TOP MATCHES ({len(opportunities)} opportunit"
        f"{'y' if len(opportunities) == 1 else 'ies'})",
        "═" * 51,
    ]

    if not opportunities:
        lines.append("   (No new matches scoring >= "
                      f"{MIN_SCORE} today.)")
    for i, opp in enumerate(opportunities, 1):
        s = opp["score"]
        d = opp["deadline"]
        c = opp["complexity"]
        days = d.get("days_left")
        days_str = f"{days} days" if days is not None else "unknown"
        lines += [
            "",
            f"#{i} — {opp['title']}",
            f"    Score: {s['overall_score']}/10  •  Deadline: "
            f"{d.get('deadline') or 'unknown'} ({days_str})",
            f"    Funding: {s.get('funding', 'unknown')}",
            f"    Why it fits: {s.get('reasoning', '')}",
            f"    Documents: {', '.join(opp['documents'].get('documents', []) or ['—'])}",
            f"    English test: {opp['documents'].get('english_test_required', 'unknown')}",
            f"    Complexity: ~{c.get('estimated_hours', '?')}h, "
            f"{c.get('difficulty', '?')} difficulty, odds: {c.get('odds', '?')}",
            f"    Recommendation: {s.get('recommendation', '—')}",
            f"    Apply: {opp['url']}",
        ]

    if opportunities:
        lines += ["", "📅 ADDED TO CALENDAR",
                  f"   {len(opportunities)} deadline(s) with reminders at 30/14/7/3/1 days"]

    if blocked_scams:
        lines += ["", "🚫 BLOCKED SCAMS (no action needed)"]
        for s in blocked_scams[:10]:
            lines.append(f"   • {s['name']} — {s['reason']}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# Single-URL analysis (transparent: shows which model ran each step)
# ════════════════════════════════════════════════════════════════════════
def analyze_url(url: str):
    cprint(f"🔎 Analyzing single URL: {url}\n")
    fetched = tools.fetch_url(url)
    if fetched["error"] or not fetched["text"]:
        cprint(f"❌ Could not fetch URL: {fetched['error'] or fetched['status']}")
        return

    cprint("[1] clean_html       → GROQ" if _has_rich() else "[1] clean_html → GROQ")
    text = tools.clean_html(fetched["text"])

    cprint("[2] first_pass_filter → GEMINI")
    filt = tools.first_pass_filter(text, PROFILE)
    cprint(f"     keep={filt['keep']} — {filt['reason']}")

    cprint("[3] check_deadline    → GEMINI")
    deadline = tools.check_deadline(text)
    cprint(f"     {deadline}")

    cprint("[4] check_legitimacy  → CLAUDE ($)")
    legit = tools.check_legitimacy(text, url)
    cprint(f"     verdict={legit['verdict']} ({legit['model_used']}) — {legit['reasoning']}")

    cprint("[5] check_eligibility → CLAUDE ($)")
    elig = tools.check_eligibility(text, PROFILE)
    cprint(f"     overall={elig['overall']} ({elig['model_used']}) — {elig['reasoning']}")
    if elig.get("blocking_issues"):
        cprint(f"     blocking: {elig['blocking_issues']}")
    if elig.get("addressable_gaps"):
        cprint(f"     gaps: {elig['addressable_gaps']}")

    cprint("[6] extract_documents → GEMINI")
    docs = tools.extract_documents(text)
    cprint(f"     {docs}")

    cprint("[7] estimate_complexity → GEMINI")
    complexity = tools.estimate_complexity(text)
    cprint(f"     {complexity}")

    cprint("[8] score_opportunity → CLAUDE ($)")
    score = tools.score_opportunity({
        "raw_text": text, "url": url, "legitimacy": legit,
        "eligibility": elig, "documents": docs,
        "complexity": complexity, "deadline": deadline,
    }, PROFILE)
    cprint(f"     SCORE = {score['overall_score']}/10 ({score['model_used']})")
    cprint(f"     {score['reasoning']}")

    daily = get_daily_summary()
    cprint(f"\n💰 This run cost: Claude=${daily['claude']['cost']:.4f} (today total)")

    if score["overall_score"] >= MIN_SCORE:
        tools.save_opportunity({"url": url, "title": fetched.get("text", "")[:60],
                                "status": "analyzed", "score": score["overall_score"]})
        cprint(f"✅ Would notify (score >= {MIN_SCORE}).")
    else:
        cprint(f"ℹ️  Below notify threshold ({MIN_SCORE}).")


# ════════════════════════════════════════════════════════════════════════
# Watchlist re-check
# ════════════════════════════════════════════════════════════════════════
def check_watchlist():
    wl = db.all_watchlist()
    if not wl:
        cprint("👀 Watchlist is empty.")
        return
    cprint(f"👀 Re-checking {len(wl)} watchlisted opportunities...")
    newly_open = []
    for oid, item in list(wl.items()):
        url = item.get("url")
        if not url:
            continue
        fetched = tools.fetch_url(url, force=True)
        if not fetched["text"]:
            continue
        deadline = tools.check_deadline(fetched["text"])
        if deadline["status"] == "open":
            newly_open.append({**item, "deadline": deadline})
            db.remove_from_watchlist(oid)
    if newly_open:
        report = build_report([], {k: 0 for k in (
            "discovered", "already_seen", "known_scam", "first_pass_dropped",
            "scam", "ineligible", "closed", "deep_analyzed", "scored_high")}, [])
        report += "\n\n👀 WATCHLIST — newly OPEN:\n" + "\n".join(
            f"   • {o.get('title','?')} — deadline {o['deadline'].get('deadline')}"
            for o in newly_open)
        tools.send_notification(report, subject="👀 OpportunityBot: watchlist items opened")
    else:
        cprint("   No watchlist items have opened yet.")


# ════════════════════════════════════════════════════════════════════════
# Cost report
# ════════════════════════════════════════════════════════════════════════
def show_cost():
    monthly = get_monthly_summary()
    daily = get_daily_summary()
    cap_d = float(os.getenv("DAILY_CLAUDE_BUDGET_USD", "0.50"))
    cap_m = float(os.getenv("MONTHLY_CLAUDE_BUDGET_USD", "10.00"))
    print("💰 OpportunityBot Cost Report")
    print("─" * 40)
    print(f"TODAY  Claude ${daily['claude']['cost']:.4f} "
          f"(cap ${cap_d:.2f})  Gemini FREE  Groq FREE")
    print(f"       calls: claude={daily['claude']['calls']} "
          f"gemini={daily['gemini']['calls']} groq={daily['groq']['calls']}")
    print("─" * 40)
    print(f"MONTH  Claude ${monthly['claude']:.4f} / ${cap_m:.2f} "
          f"({(monthly['claude']/cap_m*100 if cap_m else 0):.0f}% of budget)")
    print(f"       Gemini ${monthly['gemini']:.4f} (free)")
    print(f"       Groq   ${monthly['groq']:.4f} (free)")
    print(f"       TOTAL  ${monthly['total']:.4f}")


# ════════════════════════════════════════════════════════════════════════
# Provider smoke test
# ════════════════════════════════════════════════════════════════════════
def run_test():
    from model_router import test_providers
    print("🧪 Testing all 3 model providers...\n")
    results = test_providers()
    all_ok = True
    for name in ("claude", "gemini", "groq"):
        r = results.get(name, {})
        if r.get("ok"):
            print(f"  ✅ {name:7} OK  → {r['model']}  reply: {r['reply']!r}")
        else:
            all_ok = False
            print(f"  ❌ {name:7} FAILED — {r.get('error')}")
    print()
    if all_ok:
        print("🎉 All providers connected.")
    else:
        print("⚠️  One or more providers failed. Check API keys in .env.")
    return all_ok


# ════════════════════════════════════════════════════════════════════════
# Daemon (scheduled daily run)
# ════════════════════════════════════════════════════════════════════════
def run_daemon():
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print("❌ APScheduler not installed. pip install APScheduler")
        return
    scan_time = os.getenv("DAILY_SCAN_TIME", "08:00")
    hour, minute = (scan_time.split(":") + ["0"])[:2]
    sched = BlockingScheduler()
    sched.add_job(lambda: run_scan(), CronTrigger(hour=int(hour), minute=int(minute)))
    print(f"⏰ Daemon started. Daily scan scheduled at {scan_time}. Ctrl-C to stop.")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n👋 Daemon stopped.")


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="OpportunityBot — multi-model fellowship/scholarship agent")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan", action="store_true", help="Run a full daily scan")
    g.add_argument("--url", metavar="URL", help="Analyze one specific URL")
    g.add_argument("--watchlist", action="store_true", help="Re-check the watchlist")
    g.add_argument("--tracker", action="store_true", help="Show application tracker")
    g.add_argument("--cost", action="store_true", help="Show cost breakdown")
    g.add_argument("--daemon", action="store_true", help="Run scheduled daily")
    g.add_argument("--test", action="store_true", help="Smoke-test all 3 providers")
    args = parser.parse_args()

    if args.test:
        run_test()
    elif args.cost:
        show_cost()
    elif args.tracker:
        from tracker import show_tracker
        show_tracker()
    elif args.watchlist:
        check_watchlist()
    elif args.url:
        analyze_url(args.url)
    elif args.daemon:
        run_daemon()
    elif args.scan:
        run_scan()


if __name__ == "__main__":
    sys.exit(main())
