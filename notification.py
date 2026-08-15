"""
notification.py — dedup, notification tiers, and per-opportunity formatting.

Sits between the scan pipeline and notifier.py: notifier.py still owns the
actual email/Telegram sending, unchanged. This module decides *what* is worth
sending and *how urgently*.

  * dedupe_opportunities() collapses the same opportunity found via several
    sources into one, keeping the highest-tier source as canonical.
  * classify_tier() assigns CRITICAL / HIGH / GOOD / WATCHLIST.
  * format_opportunity() renders the full field set with an urgency emoji.
  * opportunity_of_the_day() picks the best of THAT run's real matches — it
    never invents one, and returns None when there were no matches.
"""

import re
from difflib import SequenceMatcher
from urllib.parse import urlparse

from source_whitelist import TIER_ORDER, TIER_4

# ── Notification tiers ──────────────────────────────────────────────────────
CRITICAL = "CRITICAL"   # strong match AND a deadline that is nearly here
HIGH = "HIGH"           # strong match — push now
GOOD = "GOOD"           # solid match — daily digest
WATCHLIST = "WATCHLIST"  # not open / not confirmed yet — monitor only

TIER_EMOJI = {CRITICAL: "🚨", HIGH: "🔥", GOOD: "⭐", WATCHLIST: "📌"}
TIER_ORDER_NOTIFY = [CRITICAL, HIGH, GOOD, WATCHLIST]
# Only these interrupt the user immediately.
PUSH_TIERS = (CRITICAL, HIGH)

# Deadline urgency bands (days remaining).
URGENT_DAYS = 3
SOON_DAYS = 10

# Score bands, relative to the caller's MIN_SCORE.
STRONG_MATCH_BONUS = 1.5   # MIN_SCORE + this ⇒ "strong match"
HIGH_MATCH_BONUS = 0.5

_TRACKING_PARAMS = re.compile(
    r"^(utm_[a-z]+|fbclid|gclid|mc_cid|mc_eid|ref|source|src)$", re.I)


# ── Canonicalisation ────────────────────────────────────────────────────────
def canonical_url(url: str) -> str:
    """Normalise a URL for comparison: no scheme, no www, no tracking params,
    no fragment, no trailing slash. Never raises."""
    try:
        raw = (url or "").strip()
        if not raw:
            return ""
        if "//" not in raw:
            raw = "//" + raw
        p = urlparse(raw)
        host = (p.netloc or "").lower().removeprefix("www.")
        path = (p.path or "").rstrip("/")
        kept = []
        for part in (p.query or "").split("&"):
            if not part or "=" not in part:
                continue
            key = part.split("=", 1)[0]
            if not _TRACKING_PARAMS.match(key):
                kept.append(part)
        query = "?" + "&".join(sorted(kept)) if kept else ""
        return f"{host}{path}{query}"
    except Exception:
        return (url or "").strip().lower()


def organization_of(opp: dict) -> str:
    """Best-effort organisation name from the opportunity's URL host.

    Derived, never invented — if there is no URL, we say so.
    """
    try:
        url = opp.get("official_url") or opp.get("url") or ""
        host = urlparse(url if "//" in url else "//" + url).netloc.lower()
        host = host.removeprefix("www.")
        return host or "unknown"
    except Exception:
        return "unknown"


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (title or "").lower()).strip()


def title_similarity(a: str, b: str) -> float:
    ta, tb = _title_key(a), _title_key(b)
    if not ta or not tb:
        return 0.0
    return SequenceMatcher(None, ta, tb).ratio()


def _deadline_of(opp: dict):
    d = opp.get("deadline")
    if isinstance(d, dict):
        return d.get("deadline")
    return d


def _days_left(opp: dict):
    d = opp.get("deadline")
    if isinstance(d, dict):
        return d.get("days_left")
    return None


def _score_of(opp: dict) -> float:
    s = opp.get("score") or {}
    try:
        return float(s.get("overall_score", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _tier_rank(opp: dict) -> int:
    """Lower is better. Unknown/missing tier sorts as TIER_4."""
    tier = opp.get("source_tier") or TIER_4
    try:
        return TIER_ORDER.index(tier)
    except ValueError:
        return TIER_ORDER.index(TIER_4)


# ── Dedup ───────────────────────────────────────────────────────────────────
# Cross-organisation merges (an aggregator listing an official programme) need
# a stricter title match than two pages on the same host.
CROSS_ORG_TITLE_THRESHOLD = 0.90


def _is_duplicate(a: dict, b: dict, title_threshold: float = 0.85) -> bool:
    """Same opportunity reached via different sources?"""
    ca, cb = canonical_url(a.get("url", "")), canonical_url(b.get("url", ""))
    if ca and ca == cb:
        return True

    similarity = title_similarity(a.get("title", ""), b.get("title", ""))
    da, db_ = _deadline_of(a), _deadline_of(b)
    # Different stated deadlines mean different intakes, never a duplicate.
    if da and db_ and da != db_:
        return False

    if organization_of(a) == organization_of(b):
        # Same host: a near-identical title is enough, and one side may not
        # have a deadline yet.
        return similarity >= title_threshold

    # Different hosts — typically an aggregator plus the official page. Merging
    # these is the point of source tiers, but demand stronger evidence: a very
    # close title AND the same confirmed deadline.
    return (similarity >= CROSS_ORG_TITLE_THRESHOLD
            and bool(da) and bool(db_) and da == db_)


def dedupe_opportunities(opportunities, title_threshold: float = 0.85) -> list:
    """Collapse duplicates, keeping the highest-tier source as canonical.

    Ties on tier are broken by the higher score. The surviving record gains a
    ``duplicate_sources`` list so nothing is silently discarded.
    Never raises: on unexpected input the original list is returned.
    """
    try:
        kept = [o for o in (opportunities or []) if isinstance(o, dict)]

        # Duplicate-ness is transitive: an aggregator may match the official
        # page which in turn matches another URL form of itself. Merge
        # repeatedly until a pass changes nothing, so chains fully collapse.
        for _ in range(len(kept) + 1):
            merged_any = False
            for i in range(len(kept)):
                for j in range(i + 1, len(kept)):
                    if not _is_duplicate(kept[i], kept[j], title_threshold):
                        continue
                    # Better = higher-tier source, then higher score.
                    better = (
                        (_tier_rank(kept[j]), -_score_of(kept[j]))
                        < (_tier_rank(kept[i]), -_score_of(kept[i]))
                    )
                    winner, loser = ((kept[j], kept[i]) if better
                                     else (kept[i], kept[j]))
                    dupes = list(winner.get("duplicate_sources") or [])
                    dupes += list(loser.get("duplicate_sources") or [])
                    dupes.append({"url": loser.get("url"),
                                  "source_tier": loser.get("source_tier")})
                    kept[i] = {**winner, "duplicate_sources": dupes}
                    del kept[j]
                    merged_any = True
                    break
                if merged_any:
                    break
            if not merged_any:
                break
        return kept
    except Exception as e:
        print(f"⚠️  Dedup failed ({e}) — using the undeduped list.")
        return [o for o in (opportunities or []) if isinstance(o, dict)]


# ── Tiering ─────────────────────────────────────────────────────────────────
def classify_tier(opp: dict, min_score: float = 7.0) -> str:
    """Assign a notification tier. Never raises."""
    try:
        score = _score_of(opp)
        days = _days_left(opp)
        deadline_known = _deadline_of(opp) is not None

        # Nothing confirmed to be open yet → monitor, do not interrupt.
        if not deadline_known:
            return WATCHLIST
        if score < min_score:
            return WATCHLIST

        strong = score >= min_score + STRONG_MATCH_BONUS
        if strong and days is not None and days <= SOON_DAYS:
            return CRITICAL
        if strong or (score >= min_score + HIGH_MATCH_BONUS
                      and days is not None and days <= URGENT_DAYS):
            return HIGH
        return GOOD
    except Exception:
        return WATCHLIST


def should_push(tier: str) -> bool:
    """CRITICAL/HIGH interrupt immediately; GOOD waits for the digest."""
    return tier in PUSH_TIERS


def urgency_emoji(days_left) -> str:
    if days_left is None:
        return "📅"
    try:
        d = int(days_left)
    except (TypeError, ValueError):
        return "📅"
    if d <= URGENT_DAYS:
        return "🔥"
    if d <= SOON_DAYS:
        return "⚡"
    return "📅"


# ── Formatting ──────────────────────────────────────────────────────────────
def _value_of(opp: dict) -> str:
    score = opp.get("score") or {}
    ev = score.get("expected_value") or {}
    parts = []
    reward = opp.get("reward") or score.get("funding")
    if reward and reward != "unknown":
        parts.append(str(reward))
    if opp.get("estimated_value"):
        parts.append(str(opp["estimated_value"]))
    usd = ev.get("estimated_value_usd")
    if usd:
        # Always labelled an estimate — never presented as a guaranteed payout.
        parts.append(f"~${usd:,.0f} expected (estimate)")
    return ", ".join(parts) if parts else "unknown"


def _location_of(opp: dict) -> str:
    if opp.get("remote") is True:
        return "Remote"
    loc = opp.get("location")
    if loc and opp.get("remote") is False:
        return f"{loc} (on-site)"
    return loc or "unknown"


def format_opportunity(opp: dict, index=None, min_score: float = 7.0) -> str:
    """Render one opportunity with the full field set."""
    tier = opp.get("notification_tier") or classify_tier(opp, min_score)
    score = opp.get("score") or {}
    days = _days_left(opp)
    deadline = _deadline_of(opp) or "unknown"
    days_str = f"{days} days left" if days is not None else "date unknown"
    head = f"{TIER_EMOJI.get(tier, '•')} {tier}"
    if index is not None:
        head = f"#{index} — {head}"

    lines = [
        head,
        f"    {opp.get('title', 'Untitled')}",
        f"    Category: {opp.get('category') or 'uncategorized'}"
        f"  •  Organization: {organization_of(opp)}",
        f"    Location: {_location_of(opp)}",
        f"    {urgency_emoji(days)} Deadline: {deadline} ({days_str})",
        f"    Value: {_value_of(opp)}",
        f"    Score: {score.get('overall_score', '—')}/10"
        f"  •  Credibility: {opp.get('credibility_status') or 'unknown'}"
        f"  •  Eligibility: {opp.get('eligibility_status') or 'unknown'}",
        f"    Why it matches: {score.get('reasoning', '—')}",
        f"    Apply: {opp.get('official_url') or opp.get('url', '')}",
    ]
    dupes = opp.get("duplicate_sources")
    if dupes:
        lines.append(f"    (also seen at {len(dupes)} other source"
                     f"{'s' if len(dupes) != 1 else ''})")
    return "\n".join(lines)


def group_by_tier(opportunities, min_score: float = 7.0) -> dict:
    """Assign tiers and bucket the opportunities. Returns tier -> [opps]."""
    grouped = {t: [] for t in TIER_ORDER_NOTIFY}
    for opp in opportunities or []:
        tier = classify_tier(opp, min_score)
        opp["notification_tier"] = tier
        grouped.setdefault(tier, []).append(opp)
    return grouped


def opportunity_of_the_day(opportunities, min_score: float = 7.0):
    """Best real match from THIS run, or None. Never fabricates one."""
    candidates = [o for o in (opportunities or [])
                  if isinstance(o, dict) and _score_of(o) >= min_score
                  and classify_tier(o, min_score) != WATCHLIST]
    if not candidates:
        return None
    return max(candidates, key=lambda o: (_score_of(o), -_tier_rank(o)))


def format_opportunity_of_the_day(opp) -> list:
    """Report lines for the daily pick; empty list when there is nothing."""
    if not opp:
        return []
    days = _days_left(opp)
    score = opp.get("score") or {}
    return [
        "",
        "🏆 OPPORTUNITY OF THE DAY",
        f"   {opp.get('title', 'Untitled')}",
        f"   {score.get('overall_score', '—')}/10  •  "
        f"{urgency_emoji(days)} {_deadline_of(opp) or 'deadline unknown'}  •  "
        f"{organization_of(opp)}",
        f"   {opp.get('official_url') or opp.get('url', '')}",
    ]


def build_priority_alert(opportunities, min_score: float = 7.0) -> str:
    """Short message for the CRITICAL/HIGH push. Empty string if none."""
    urgent = [o for o in (opportunities or [])
              if should_push(o.get("notification_tier")
                             or classify_tier(o, min_score))]
    if not urgent:
        return ""
    lines = [f"🚨 {len(urgent)} time-sensitive opportunit"
             f"{'y' if len(urgent) == 1 else 'ies'}", ""]
    for i, opp in enumerate(urgent, 1):
        lines.append(format_opportunity(opp, i, min_score))
        lines.append("")
    return "\n".join(lines).rstrip()
