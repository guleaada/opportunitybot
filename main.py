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
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # load .env before importing modules that read os.getenv at import

import tools
import taxonomy
import discovery
import prioritizer
import notification
from model_router import ProvidersUnavailable
from profile import PROFILE
from checker import (
    meets_threshold, PROBABLY_ELIGIBLE, normalize_credibility, is_notifiable,
    HIGH_RISK, SUSPICIOUS,
)
from source_whitelist import load_whitelist, domain_quality, source_tier
from cost_tracker import (
    get_daily_summary, get_monthly_summary, get_daily_claude_spend,
    get_monthly_claude_spend,
)
import database as db
from ci_persistence import is_running_in_ci, ensure_data_dir, log_run_metadata

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
# Shortest feed snippet worth analyzing when the full page fetch is blocked.
MIN_SNIPPET_CHARS = int(os.getenv("MIN_SNIPPET_CHARS", "80"))
# How much RSS body text to keep. This is the primary analysis input whenever
# the direct and reader fetches are blocked, so it is a body, not a teaser.
RSS_SNIPPET_CHARS = int(os.getenv("RSS_SNIPPET_CHARS", "2000"))

# Discovery feeds. Unreachable or moved feeds are logged and skipped per-feed;
# the per-feed "📰 RSS <domain>: +N posts" line shows which are productive.
RSS_FEEDS = list(dict.fromkeys([   # dict.fromkeys keeps order AND dedupes
    # ── Confirmed live (returned +10 posts each last run) ────────────────
    # These skew to student scholarships, which the eligibility check
    # correctly rejects for a working professional — kept for coverage, but
    # they are not where matches are expected to come from.
    "https://opportunitiescorners.com/feed/",
    "https://www.opportunitiesforafricans.com/feed/",
    "https://opportunitydesk.org/feed/",
    # ── Remote / developer jobs — corrected paths ───────────────────────
    # The profile is a working AI professional, so these are the feeds most
    # likely to yield candidates that survive eligibility.
    "https://weworkremotely.com/remote-jobs.rss",
    "https://www.python.org/jobs/feed/rss/",
    "https://stackoverflow.com/jobs/feed",
    # ── Professional aggregators (not student-only) ─────────────────────
    "https://www.opportunitiesforyouth.org/feed/",
    # Removed as permanently dead — they failed every run and cost a DNS or
    # connection timeout each time:
    #   jobs.github.com                — GitHub Jobs retired in 2021 (refused)
    #   remoteok.com                   — HTTP 410 Gone
    #   opportunitiesforyoungpeople.com — DNS NXDOMAIN
]))

# Default taxonomy category per feed domain, so job items are tagged rather
# than "uncategorized". Purely descriptive — it feeds the report and the
# stored record, NOT the scoring math.
FEED_CATEGORY = {
    "weworkremotely.com": "remote_jobs",
    "www.python.org": "dev_jobs",
    "stackoverflow.com": "dev_jobs",
}

# The RSS feeds that are NOT tagged as job boards are the fellowship /
# scholarship / grant publishers. Derived rather than listed twice, so adding
# a feed to RSS_FEEDS automatically classifies it.
OPPORTUNITY_FEED_DOMAINS = frozenset(
    f.split("/")[2] for f in RSS_FEEDS
    if len(f.split("/")) > 2 and f.split("/")[2] not in FEED_CATEGORY
)


def _search_provider():
    """Pick the web-search backend for this scan.

    Tavily first when its key is configured, because Google's Custom Search
    JSON API is unavailable to this project (403 PERMISSION_DENIED — the Cloud
    project has no access to that API). Google is kept intact and is used
    whenever its credentials are present and Tavily's are not, so restoring
    CSE later is a matter of configuration, not code.

    With neither configured a Google-shaped provider is still returned: it
    reports DISABLED with the missing variable names, exactly as before.
    """
    if os.getenv("TAVILY_API_KEY"):
        import tavily_search
        return discovery.SearchProvider(
            search_fn=tavily_search.tavily_search,
            quality_fn=domain_quality,
            name="tavily",
            required_env=("TAVILY_API_KEY",),
            state=tavily_search,
        )
    return discovery.SearchProvider(search_fn=tools.web_search,
                                    quality_fn=domain_quality)


# ════════════════════════════════════════════════════════════════════════
# Core pipeline
# ════════════════════════════════════════════════════════════════════════
def analyze_one(result, stats: dict):
    """Run the full triage pipeline on a single search result.

    Returns a rich dict if it's a notify-worthy match, else None. Mutates
    ``stats`` counters along the way.

    If every model provider is unavailable, the candidate is marked
    ANALYSIS_UNAVAILABLE and PRESERVED for a later run — a technical outage is
    never treated as ineligibility.
    """
    try:
        return _analyze_one_inner(result, stats)
    except ProvidersUnavailable as e:
        url = getattr(result, "url", None) or ""
        title = getattr(result, "title", None) or ""
        stats["analysis_unavailable"] = stats.get("analysis_unavailable", 0) + 1
        cprint(f"   🟡 ANALYSIS_UNAVAILABLE (all providers down) — "
               f"preserving candidate for a later run: {e}")
        # Deliberately NOT save_opportunity(): that would mark it seen and it
        # would never be retried. The watchlist is the existing retry channel.
        try:
            tools.add_to_watchlist({
                "url": url, "title": title,
                "reason": "ANALYSIS_UNAVAILABLE — all model providers were "
                          "unavailable; retry when quota resets"})
        except Exception as werr:
            db.log_error(f"Could not preserve {url}: {werr}")
        return None


def _analyze_one_inner(result, stats: dict):
    # RSS entries frequently omit fields — coerce None to "" so a missing
    # title or snippet can never crash this candidate (or the whole scan).
    url = getattr(result, "url", None) or ""
    title = getattr(result, "title", None) or ""
    snippet = getattr(result, "snippet", None) or ""
    cprint(f"   ▶️ analyze_one ENTER: {url[:60] if url else '(no url)'}")
    if not url:
        stats["fetch_failed"] += 1
        cprint("   ⚠️  candidate has no URL — skipping")
        return None

    # 4. fetch (no model). fetch_url is contracted to return an error dict, but
    # if anything ever raises out of it the exception would escape before a
    # single counter moved — the all-zeros signature. Route a raise into the
    # same snippet fallback instead of losing the candidate.
    try:
        fetched = tools.fetch_url(url)
    except Exception as e:
        cprint(f"   ⚠️  fetch_url raised ({type(e).__name__}: {e}) — "
               f"treating as a failed fetch")
        fetched = {"url": url, "html": "", "text": "", "status": 0,
                   "cached": False, "error": f"{type(e).__name__}: {e}"}

    if fetched["error"] or fetched["status"] != 200 or not fetched["text"]:
        reason = fetched["error"] or fetched["status"]
        db.log_error(f"Fetch failed for {url}: {reason}")

        # Tier 2 — reader proxy. Jina fetches server-side, so this runner's IP
        # being Cloudflare-blocked doesn't matter, and we get the FULL page
        # rather than a short feed snippet (which is too thin for the deadline
        # and eligibility checks to confirm anything).
        try:
            jina = tools.fetch_via_jina(url)
        except Exception as e:
            jina = {"text": "", "status": 0,
                    "error": f"{type(e).__name__}: {e}"}

        if (jina.get("text") or "").strip():
            text = jina["text"]
            stats["jina_fetch"] = stats.get("jina_fetch", 0) + 1
            cprint(f"   ↩️  direct fetch blocked ({reason}) — got "
                   f"{len(text)} chars of full text via jina reader")
        # Tier 3 — the feed's own snippet, only if the reader failed too.
        elif len(snippet) >= MIN_SNIPPET_CHARS:
            text = f"{title}\n\n{snippet}"
            stats["snippet_fallback"] = stats.get("snippet_fallback", 0) + 1
            cprint(f"   ↩️  fetch + jina blocked ({reason}) — analyzing the "
                   f"{len(snippet)}-char feed snippet instead")
        else:
            stats["fetch_failed"] += 1
            cprint(f"   ⚠️  fetch + jina failed ({reason}) and no usable "
                   f"snippet ({len(snippet)} chars) — skipping")
            return None
    else:
        # 5. clean_html (LOCAL, no model) — deterministic text normalization
        text = tools.clean_html(fetched["text"])

    # 6. first_pass_filter (GEMINI free) — drops most candidates
    filt = tools.first_pass_filter(text, PROFILE)
    if not filt["keep"]:
        # The first-pass filter is tuned for scholarships, so it drops
        # bounties, prize pools and open calls that never use that vocabulary.
        # Give those a second look before discarding (free phrase scan; the
        # cheap model only runs on genuinely ambiguous pages).
        signal = tools.detect_opportunity(text, title)
        if signal.get("is_opportunity"):
            stats["signal_rescued"] = stats.get("signal_rescued", 0) + 1
            cprint(f"   🕵️  rescued by signals ({signal.get('method')}): "
                   f"{signal.get('reason', '')}")
        else:
            stats["first_pass_dropped"] += 1
            tools.save_opportunity({"url": url, "title": title,
                                    "status": "filtered_out", "reason": filt["reason"]})
            cprint(f"   ✂️  dropped: {filt['reason']}")
            return None

    # 7. check_deadline (GEMINI free) — gate BEFORE any paid Claude call.
    # A closed program must not burn legitimacy/eligibility tokens. Closed
    # programs that passed the first-pass filter are usually annual — put
    # them on the watchlist so --watchlist re-checks when they reopen.
    deadline = tools.check_deadline(text)
    if deadline["status"] == "closed":
        stats["closed"] += 1
        tools.save_opportunity({"url": url, "title": title, "status": "closed",
                                "deadline": deadline.get("deadline")})
        tools.add_to_watchlist({"url": url, "title": title,
                                "last_known_deadline": deadline.get("deadline"),
                                "reason": "closed at discovery; likely annual"})
        cprint("   📕 deadline closed — added to watchlist")
        return None

    # 8. check_legitimacy (CLAUDE $) — only on open survivors
    legit = tools.check_legitimacy(text, url)
    cred = legit.get("credibility_status") or normalize_credibility(legit["verdict"])
    if cred in (HIGH_RISK, SUSPICIOUS):
        stats["scam"] += 1
        tools.save_opportunity({"url": url, "title": title, "status": legit["verdict"],
                                "credibility_status": cred,
                                "source_tier": legit.get("source_tier"),
                                "reasoning": legit["reasoning"]})
        cprint(f"   🚫 {cred}: {legit['reasoning']}")
        return None
    if not is_notifiable(cred):
        # NEEDS_VERIFICATION — unfamiliar, not accused. Honesty rule: we do not
        # recommend what we could not verify.
        stats["legit_unknown"] += 1
        tools.save_opportunity({"url": url, "title": title,
                                "status": "legitimacy_unknown",
                                "credibility_status": cred,
                                "source_tier": legit.get("source_tier"),
                                "reasoning": legit["reasoning"]})
        cprint(f"   ❓ {cred} (tier {legit.get('source_tier')}) — "
               f"skipping (honesty rule)")
        return None

    # 9. check_eligibility (CLAUDE $) — graded ladder; anything below
    # PROBABLY_ELIGIBLE (including UNCERTAIN) never reaches notification.
    elig = tools.check_eligibility(text, PROFILE)
    elig_status = elig.get("eligibility_status", "UNCERTAIN")
    if not meets_threshold(elig_status, PROBABLY_ELIGIBLE):
        stats["ineligible"] += 1
        tools.save_opportunity({"url": url, "title": title,
                                "status": elig["overall"],
                                "eligibility_status": elig_status,
                                "reasoning": elig["reasoning"]})
        cprint(f"   ⛔ {elig_status}: {elig['reasoning']}")
        return None

    # 10. enrichment (GEMINI free)
    docs = tools.extract_documents(text)
    complexity = tools.estimate_complexity(text)

    # 11. score_opportunity (CLAUDE $)
    score = tools.score_opportunity({
        "raw_text": text, "url": url,
        "legitimacy": legit, "eligibility": elig,
        "documents": docs, "complexity": complexity, "deadline": deadline,
    }, PROFILE)
    stats["deep_analyzed"] += 1

    # Surface the expected-value sub-scores in the run log.
    subs = score.get("sub_scores") or {}
    if subs:
        ev = score.get("expected_value") or {}
        prob = score.get("probability") or {}
        cprint("   📐 " + "  ".join(
            f"{k.replace('_score', '')}={'—' if v is None else v}"
            for k, v in subs.items()))
        cprint(f"   🎲 win probability ≈ {prob.get('probability')} "
               f"(confidence: {prob.get('confidence')})  •  "
               f"expected value {ev.get('expected_value_score')}/10 "
               f"→ final {score.get('final_score')}/10")

    # Richer data model — every field present, None/unknown when not derivable.
    enrichment = {
        "category": getattr(result, "category", None),
        "official_url": url,
        "location": getattr(result, "location", None),
        "remote": getattr(result, "remote", None),
        "deadline": deadline.get("deadline"),
        "reward": score.get("funding"),
        "estimated_value": getattr(result, "estimated_value", None),
        "eligibility_status": elig_status,
        "credibility_status": cred,
        "source_tier": legit.get("source_tier") or source_tier(url),
        "source_quality": domain_quality(url) or getattr(result, "source_quality", None),
        "requirements": docs.get("documents", []) or [],
    }

    tools.save_opportunity({
        "url": url, "title": title, "status": "analyzed",
        "score": score["overall_score"], **enrichment,
    })

    if score["overall_score"] >= MIN_SCORE:
        stats["scored_high"] += 1
        # enrichment first: the explicit keys below win, so ``deadline`` stays
        # the full analysis dict that build_report expects.
        return {
            **enrichment,
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

    # Per-scan model provider state: reset counters/breaker, log config.
    try:
        import model_router
        model_router.reset_provider_stats()
        model_router.log_model_configuration()
    except Exception as e:
        cprint(f"⚠️ Could not initialise model providers: {e}")

    # Per-scan budget for the free hidden-opportunity classifier.
    try:
        import signals
        signals.reset_classification_budget()
    except Exception as e:
        cprint(f"⚠️ Could not reset classification budget: {e}")

    # --- RSS discovery (primary source while CSE is down) -------------------
    def _fetch_rss_feeds(ctx=None):
        import html as _html
        import urllib.request
        import xml.etree.ElementTree as ET
        from search import SearchResult

        # WordPress puts the FULL post body in content:encoded; <description>
        # is usually a truncated teaser. Since the direct and reader fetches
        # are both blocked from this runner, this text IS the analysis input.
        CONTENT_NS = "{http://purl.org/rss/1.0/modules/content/}encoded"

        def _plain(el):
            """Element -> plain text: all descendant text, entities decoded,
            tags stripped, whitespace collapsed. '' for a missing element."""
            if el is None:
                return ""
            raw = "".join(el.itertext())          # handles CDATA and children
            raw = _html.unescape(raw)             # &amp;/&#8217;/&nbsp; -> chars
            raw = re.sub(r"<[^>]+>", " ", raw)    # strip any embedded markup
            return re.sub(r"\s+", " ", raw).strip()

        import ssl as _ssl
        import urllib.error

        health = (ctx or {}).get("health") if ctx else None
        out, ok_feeds, failed, skipped = [], 0, 0, 0

        for feed_url in RSS_FEEDS[:discovery.RSS_MAX_FEEDS_PER_SCAN]:
            domain = feed_url.split("/")[2]
            key = f"rss:{domain}"
            # A feed that has 404'd repeatedly is gone — stop requesting it.
            if health is not None and discovery.is_disabled(health, key):
                skipped += 1
                cprint(f"⏭️  RSS {domain}: skipped (permanently unavailable)")
                continue
            try:
                req = urllib.request.Request(
                    feed_url,
                    headers={"User-Agent":
                             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) "
                             "Chrome/125.0.0.0 Safari/537.36"})
                with urllib.request.urlopen(req, timeout=25) as resp:
                    raw = resp.read()
                root = ET.fromstring(raw)
                items = root.findall(".//item")
                count = 0
                for it in items[:20]:
                    title = _plain(it.find("title"))
                    link = _plain(it.find("link"))
                    # Prefer the full body; fall back to the teaser.
                    body = _plain(it.find(CONTENT_NS)) or _plain(it.find("description"))
                    body = body[:RSS_SNIPPET_CHARS]
                    if not title or not link:
                        continue
                    out.append(SearchResult(title=title, url=link,
                                            snippet=body, source=domain,
                                            category=FEED_CATEGORY.get(domain)))
                    count += 1
                ok_feeds += 1
                if health is not None:
                    discovery.record_success(health, key, count)
                cprint(f"📰 RSS {domain}: +{count} posts")
            except Exception as e:
                # A dead or moved feed contributes 0 and never stops the scan.
                # Classify so permanent 404s can be retired and transient
                # 403/SSL/timeout failures are only paused.
                failed += 1
                status = getattr(e, "code", None)
                if isinstance(e, urllib.error.HTTPError) and status == 404:
                    kind = "404"
                elif isinstance(e, urllib.error.HTTPError) and status == 403:
                    kind = "403"
                elif isinstance(e, urllib.error.HTTPError) and status == 429:
                    kind = "429"
                elif isinstance(e, _ssl.SSLError) or "SSL" in str(e).upper():
                    kind = "ssl"
                elif isinstance(e, TimeoutError) or "timed out" in str(e).lower():
                    kind = "timeout"
                else:
                    kind = "error"
                if health is not None:
                    discovery.record_failure(health, key, kind, status)
                cprint(f"⚠️ RSS feed {domain} failed [{kind}]: {e}")

        avg = int(sum(len(r.snippet or '') for r in out) / len(out)) if out else 0
        cprint(f"📰 RSS total: {len(out)} posts from {ok_feeds}/{len(RSS_FEEDS)} "
               f"feeds (avg {avg} chars of text per post)")
        return out, {"attempted": ok_feeds + failed, "successful": ok_feeds,
                     "failed": failed, "skipped_dead": skipped}
    # ------------------------------------------------------------------------

    # 1. search across whitelisted sources + taxonomy families (no model)
    sources = load_whitelist()

    # Provider-based discovery: Google is budgeted + rotated + circuit-broken,
    # and is no longer the only path. Each provider is isolated; the rest still
    # contribute if one fails. Everything downstream is unchanged.
    from source_whitelist import all_seed_urls

    providers = [
        _search_provider(),
        discovery.RSSProvider(fetch_fn=_fetch_rss_feeds),
        discovery.APIProvider(),
        discovery.SeedProvider(seeds_fn=all_seed_urls),
    ]
    all_results, discovery_summary = discovery.run_discovery(
        providers, per_query=max_results_per_source)
    seen_urls = {r.url for r in all_results if r.url}

    stats["discovery"] = discovery_summary
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

    for line in discovery.format_summary(discovery_summary,
                                         new_candidates=len(candidates)):
        cprint(line)
    cprint(f"📊 Found {stats['discovered']} total, {len(candidates)} new candidates "
           f"({stats['already_seen']} seen, {stats['known_scam']} blocked)")

    # 3b. Rank before the MAX_PER_DAY slice (local scoring, no model/network).
    # Discovery returns provider order, so the fast job APIs used to take every
    # analysis slot while the fellowship feeds got the leftovers. Nothing is
    # dropped here — low scorers stay in the pool, they just rank below the cut.
    try:
        ranked = prioritizer.rank(
            candidates,
            job_feed_domains=set(FEED_CATEGORY),
            opportunity_feed_domains=OPPORTUNITY_FEED_DOMAINS)
        candidates = [r for _, r in ranked]
        for line in prioritizer.top_lines(ranked):
            cprint(line)
    except Exception as e:
        # Ranking is an optimisation; never let it cost us a scan.
        cprint(f"⚠️  Candidate prioritisation failed ({e}) — using discovery order.")

    # 4-11. deep pipeline (budget-bounded by MAX_PER_DAY)
    cprint(f"🧮 MAX_PER_DAY={MAX_PER_DAY} — analyzing "
           f"{len(candidates[:MAX_PER_DAY])} of {len(candidates)} candidates")
    final_opportunities = []
    for r in candidates[:MAX_PER_DAY]:
        # RSS items routinely arrive with a missing title; slicing None here
        # (outside the try) used to abort the whole loop, not just this item.
        cprint(f"🔎 {(getattr(r, 'title', None) or '(untitled)')[:70]}")
        try:
            match = analyze_one(r, stats)
            if match:
                final_opportunities.append(match)
        except Exception as e:
            import traceback
            cprint(f"   ❌ analyze_one failed for {getattr(r, 'url', '?')}: {e}")
            cprint(traceback.format_exc())
            db.log_error(f"Failed to analyze {getattr(r, 'url', '?')}: {e}")
            continue

    # 12. dedupe across sources, tier, report + notify
    final_opportunities.sort(key=lambda m: m["score"]["overall_score"], reverse=True)
    before_dedupe = len(final_opportunities)
    final_opportunities = notification.dedupe_opportunities(final_opportunities)
    if before_dedupe != len(final_opportunities):
        cprint(f"🧹 Deduped {before_dedupe - len(final_opportunities)} duplicate "
               f"opportunit{'y' if before_dedupe - len(final_opportunities) == 1 else 'ies'} "
               f"(kept the highest-tier source)")
    stats["deduped"] = before_dedupe - len(final_opportunities)

    # Assign notification tiers (mutates each opp with notification_tier).
    grouped = notification.group_by_tier(final_opportunities, MIN_SCORE)
    for tier, items in grouped.items():
        if items:
            cprint(f"{notification.TIER_EMOJI.get(tier, '•')} {tier}: {len(items)}")

    # Anything not confirmed open goes to the existing watchlist, not a push.
    for opp in grouped.get(notification.WATCHLIST, []):
        try:
            tools.add_to_watchlist({
                "url": opp.get("url"), "title": opp.get("title"),
                "last_known_deadline": (opp.get("deadline") or {}).get("deadline")
                if isinstance(opp.get("deadline"), dict) else opp.get("deadline"),
                "reason": "monitoring — not confirmed open at scan time"})
        except Exception as e:
            db.log_error(f"Watchlist add failed for {opp.get('url')}: {e}")

    report = build_report(final_opportunities, stats, blocked_scams)
    top_name = final_opportunities[0]["title"] if final_opportunities else ""
    subject = tools.generate_email_subject(len(final_opportunities), top_name)

    # CRITICAL/HIGH interrupt immediately; GOOD rides the digest below.
    alert = notification.build_priority_alert(final_opportunities, MIN_SCORE)
    if alert:
        try:
            tools.send_notification(alert, channel="all",
                                    subject=f"🚨 {subject}")
        except Exception as e:
            db.log_error(f"Priority alert failed: {e}")

    tools.send_notification(report, channel="all", subject=subject)

    # Record what we notified about, so nothing is announced twice.
    for opp in final_opportunities:
        try:
            tools.save_opportunity({
                "url": opp.get("url"), "title": opp.get("title"),
                "status": "notified",
                "notification_tier": opp.get("notification_tier"),
                "notified_at": datetime.now(timezone.utc).isoformat()})
        except Exception as e:
            db.log_error(f"Could not mark notified for {opp.get('url')}: {e}")

    # calendar
    for opp in final_opportunities:
        try:
            tools.add_to_calendar({"title": opp["title"], "url": opp["url"],
                                   "deadline": opp["deadline"].get("deadline")})
        except Exception as e:
            db.log_error(f"Calendar add failed for {opp['url']}: {e}")

    db.log_activity("scan_complete", "daily scan finished",
                    matches=len(final_opportunities), stats=stats)
    log_run_metadata("scan", {**stats, "matches": len(final_opportunities)})

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
        f"   Fetch failed:          {stats.get('fetch_failed', 0)}",
        f"   Jina fetch:            {stats.get('jina_fetch', 0)}",
        f"   Snippet fallback:      {stats.get('snippet_fallback', 0)}",
        f"   First-pass filtered:   {stats['first_pass_dropped']}",
        f"   Flagged scam (Claude): {stats['scam']}",
        f"   Needs verification:    {stats.get('legit_unknown', 0)}",
        f"   Ineligible:            {stats['ineligible']}",
        f"   Closed deadline:       {stats['closed']}",
        f"   Deep-analyzed (Claude):{stats['deep_analyzed']}",
        f"   Scoring >= {MIN_SCORE}:         {stats['scored_high']}",
        f"   Rescued by signals:    {stats.get('signal_rescued', 0)}",
        f"   Analysis unavailable:  {stats.get('analysis_unavailable', 0)}",
        f"   Deduped duplicates:    {stats.get('deduped', 0)}",
        "",
        "💰 COST TODAY",
        f"   Claude: ${daily['claude']['cost']:.3f} ({daily['claude']['calls']} calls)",
        f"   Gemini: $0.00 ({daily['gemini']['calls']} free calls)",
        f"   Groq:   $0.00 ({daily['groq']['calls']} free calls)",
        f"   OpenRtr:$0.00 ({daily.get('openrouter', {}).get('calls', 0)} free calls)",
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

    # Opportunity of the Day — the best REAL match from this run only.
    try:
        import model_router
        lines += [""] + model_router.model_health_lines()
    except Exception:
        pass

    lines += notification.format_opportunity_of_the_day(
        notification.opportunity_of_the_day(opportunities, MIN_SCORE))

    # Tiered sections: urgent first, monitor-only last.
    grouped = notification.group_by_tier(opportunities, MIN_SCORE)
    counter = 0
    for tier in notification.TIER_ORDER_NOTIFY:
        items = grouped.get(tier) or []
        if not items:
            continue
        emoji = notification.TIER_EMOJI.get(tier, "•")
        note = {
            notification.CRITICAL: "act now — deadline is close",
            notification.HIGH: "strong match — pushed immediately",
            notification.GOOD: "solid match — daily digest",
            notification.WATCHLIST: "monitoring only — not confirmed open",
        }.get(tier, "")
        lines += ["", f"{emoji} {tier} ({len(items)}) — {note}", "─" * 51]
        for opp in items:
            counter += 1
            lines.append("")
            lines.append(notification.format_opportunity(opp, counter, MIN_SCORE))
            # Keep the application-effort detail the old report carried.
            c = opp.get("complexity") or {}
            docs = opp.get("documents") or {}
            lines.append(
                f"    Documents: "
                f"{', '.join(docs.get('documents', []) or ['—'])}"
                f"  •  English test: {docs.get('english_test_required', 'unknown')}")
            lines.append(
                f"    Complexity: ~{c.get('estimated_hours', '?')}h, "
                f"{c.get('difficulty', '?')} difficulty, odds: {c.get('odds', '?')}"
                f"  •  Recommendation: "
                f"{(opp.get('score') or {}).get('recommendation', '—')}")

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

    cprint("[1] clean_html       → LOCAL (no model call)")
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

    title = _page_title(fetched["html"]) or url
    tools.save_opportunity({"url": url, "title": title, "status": "analyzed",
                            "score": score["overall_score"],
                            "deadline": deadline.get("deadline")})
    if score["overall_score"] >= MIN_SCORE:
        cprint(f"✅ Would notify (score >= {MIN_SCORE}).")
    else:
        cprint(f"ℹ️  Below notify threshold ({MIN_SCORE}).")


def _page_title(html: str) -> str:
    """Pull <title> from raw HTML (no model call)."""
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup
        tag = BeautifulSoup(html, "html.parser").find("title")
        return tag.get_text().strip()[:120] if tag else ""
    except Exception:
        return ""


# ════════════════════════════════════════════════════════════════════════
# Cover-letter draft (Gemini, free)
# ════════════════════════════════════════════════════════════════════════
def draft_cover_letter(url: str):
    """Fetch an opportunity page and write a motivation-letter draft to disk."""
    cprint(f"✍️  Drafting cover letter for: {url}\n")
    fetched = tools.fetch_url(url)
    if fetched["error"] or not fetched["text"]:
        cprint(f"❌ Could not fetch URL: {fetched['error'] or fetched['status']}")
        return

    text = tools.clean_html(fetched["text"])
    title = _page_title(fetched["html"]) or url
    docs = tools.extract_documents(text)
    result = tools.generate_cover_letter(
        {"title": title, "raw_text": text, "documents": docs}, PROFILE)

    drafts_dir = Path("drafts")
    drafts_dir.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "draft"
    out_path = drafts_dir / f"{slug}.txt"
    out_path.write_text(
        f"DRAFT cover letter — review and personalize before sending!\n"
        f"Opportunity: {title}\nURL: {url}\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')} "
        f"by {result['model_used']}\n"
        + "─" * 60 + "\n\n" + result["draft"] + "\n")
    cprint(f"📝 Draft saved to {out_path}")
    cprint("─" * 60)
    cprint(result["draft"])


# ════════════════════════════════════════════════════════════════════════
# Watchlist re-check
# ════════════════════════════════════════════════════════════════════════
def check_watchlist():
    wl = db.all_watchlist()
    if not wl:
        cprint("👀 Watchlist is empty.")
        return
    cprint(f"👀 Re-checking {len(wl)} watchlisted opportunities...")
    from search import SearchResult
    stats = {k: 0 for k in (
        "discovered", "already_seen", "known_scam", "fetch_failed",
        "first_pass_dropped", "scam", "legit_unknown", "ineligible", "closed",
        "deep_analyzed", "scored_high")}
    matches, opened = [], 0
    for oid, item in list(wl.items()):
        url = item.get("url")
        if not url:
            continue
        fetched = tools.fetch_url(url, force=True)
        if not fetched["text"]:
            continue
        deadline = tools.check_deadline(fetched["text"])
        if deadline["status"] != "open":
            continue
        # It reopened — pull it off the watchlist and run the FULL pipeline
        # so it gets the same legitimacy/eligibility/scoring as scan finds.
        opened += 1
        db.remove_from_watchlist(oid)
        cprint(f"   🔓 reopened: {item.get('title', url)}")
        try:
            match = analyze_one(
                SearchResult(title=item.get("title", url), url=url), stats)
            if match:
                matches.append(match)
        except Exception as e:
            db.log_error(f"Watchlist analysis failed for {url}: {e}")

    if opened:
        matches.sort(key=lambda m: m["score"]["overall_score"], reverse=True)
        report = build_report(matches, {**stats, "discovered": opened}, [])
        report = report.replace("Daily Report", "Watchlist Report", 1)
        tools.send_notification(
            report, subject=f"👀 OpportunityBot: {opened} watchlist item(s) reopened")
        log_run_metadata("watchlist", {**stats, "reopened": opened,
                                       "matches": len(matches)})
    else:
        cprint("   No watchlist items have opened yet.")
        log_run_metadata("watchlist", {"reopened": 0})


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
    g.add_argument("--draft", metavar="URL",
                   help="Generate a cover-letter draft for an opportunity URL")
    g.add_argument("--watchlist", action="store_true", help="Re-check the watchlist")
    g.add_argument("--tracker", action="store_true", help="Show application tracker")
    g.add_argument("--cost", action="store_true", help="Show cost breakdown")
    g.add_argument("--daemon", action="store_true", help="Run scheduled daily")
    g.add_argument("--test", action="store_true", help="Smoke-test all 3 providers")
    args = parser.parse_args()

    # Initialize data files (with correct shape) before anything touches them.
    ensure_data_dir()
    cprint(f"🤖 OpportunityBot — {datetime.now().isoformat(timespec='seconds')}")
    cprint(f"   Environment: {'GitHub Actions' if is_running_in_ci() else 'Local'}")

    if args.test:
        # Non-zero exit on failure so the CI `test` mode actually fails red.
        return 0 if run_test() else 1
    elif args.cost:
        show_cost()
    elif args.tracker:
        from tracker import show_tracker
        show_tracker()
    elif args.watchlist:
        check_watchlist()
    elif args.url:
        analyze_url(args.url)
    elif args.draft:
        draft_cover_letter(args.draft)
    elif args.daemon:
        run_daemon()
    elif args.scan:
        run_scan()
    return 0


if __name__ == "__main__":
    sys.exit(main())
