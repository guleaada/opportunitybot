"""
tools.py — the agent's tool surface.

Each tool either (a) delegates to model_router.call_model() with the correct
``task_type`` so routing is automatic, or (b) is purely mechanical (no model).
This module is the single import the pipeline and CLI use, so model routing
stays consistent everywhere.

Routing summary:
  clean_html / extract_text / translate  → groq   (mechanical, free)
  check_deadline / first_pass_filter      → gemini (free)
  extract_documents / estimate_complexity → gemini (free)
  generate_cover_letter / email subject   → gemini (free)
  check_legitimacy / check_eligibility     → claude ($ high stakes)
  score_opportunity                        → claude ($ high stakes)
"""

from typing import List

from model_router import call_model, extract_json
from search import web_search as _web_search, fetch_url as _fetch_url, fetch_rss_feeds as _fetch_rss_feeds, SearchResult
from known_scams import check_known_scam as _check_known_scam
from checker import (
    check_deadline as _check_deadline,
    first_pass_filter as _first_pass_filter,
    check_legitimacy as _check_legitimacy,
    check_eligibility as _check_eligibility,
)
from scorer import score_opportunity as _score_opportunity
from cover_letter import generate_cover_letter as _generate_cover_letter
import database as db

_MAX_TEXT = 14000


def _trim(text, limit=_MAX_TEXT):
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]..."


# ── 1. web_search (no model) ────────────────────────────────────────────────
def web_search(query: str, max_results: int = 10) -> List[SearchResult]:
    return _web_search(query, max_results)


# ── 2. fetch_url (no model) ─────────────────────────────────────────────────
def fetch_url(url: str, force: bool = False) -> dict:
    return _fetch_url(url, force=force)


# ── 2b. fetch_rss_feeds (no model) ──────────────────────────────────────────
def fetch_rss_feeds(feeds=None, max_items_per_feed: int = 20) -> List[SearchResult]:
    return _fetch_rss_feeds(feeds, max_items_per_feed)


# ── 3. clean_html (GROQ, mechanical) ────────────────────────────────────────
def clean_html(html_or_text: str) -> str:
    """Convert messy HTML/text into clean readable plain text via Groq.

    Falls back to returning the input unchanged if the model call fails. The
    input is usually already BeautifulSoup-extracted by fetch_url(), so this is
    a polish step, not the primary extractor.
    """
    if not html_or_text:
        return ""
    system = (
        "You convert raw web page text into clean, readable plain text. "
        "Remove navigation menus, cookie banners, ads, and boilerplate. "
        "Keep the substantive content about the program: eligibility, funding, "
        "deadlines, requirements. Do not summarize or add commentary."
    )
    prompt = f"Clean this page text:\n\n{_trim(html_or_text)}"
    try:
        res = call_model("clean_html", prompt, system=system, max_tokens=3000)
        return (res["content"] or html_or_text).strip()
    except Exception as e:
        print(f"⚠️  clean_html failed, using raw text: {e}")
        return html_or_text


# ── 4. check_deadline (GEMINI) ──────────────────────────────────────────────
def check_deadline(text: str) -> dict:
    return _check_deadline(text)


# ── 5. first_pass_filter (GEMINI) ───────────────────────────────────────────
def first_pass_filter(text: str, profile: dict) -> dict:
    return _first_pass_filter(text, profile)


# ── 6. check_legitimacy (CLAUDE) ────────────────────────────────────────────
def check_legitimacy(text: str, source_url: str) -> dict:
    return _check_legitimacy(text, source_url)


# ── 7. check_eligibility (CLAUDE) ───────────────────────────────────────────
def check_eligibility(text: str, profile: dict) -> dict:
    return _check_eligibility(text, profile)


# ── 8. extract_documents (GEMINI) ───────────────────────────────────────────
def extract_documents(text: str) -> dict:
    """Extract required application documents. Returns a structured dict."""
    system = (
        "You extract the list of required application documents from a "
        "scholarship/fellowship page. If something is not stated, omit it — "
        "do not invent requirements."
    )
    prompt = (
        "List the required documents. Reply ONLY with JSON:\n"
        '{"documents": ["..."], "english_test_required": '
        '"IELTS|TOEFL|Duolingo|none|unknown", '
        '"references_required": number or null, '
        '"transcripts_required": true/false, '
        '"notes": "anything important about documents"}\n\n'
        f"TEXT:\n{_trim(text)}"
    )
    res = call_model("extract_document_requirements", prompt, system=system, max_tokens=500)
    return extract_json(res["content"]) or {
        "documents": [], "english_test_required": "unknown",
        "references_required": None, "transcripts_required": False,
        "notes": "extraction failed",
    }


# ── 9. estimate_complexity (GEMINI) ─────────────────────────────────────────
def estimate_complexity(text: str) -> dict:
    """Estimate effort + odds. Returns hours, difficulty, odds."""
    system = (
        "You estimate the effort required to apply to a scholarship/fellowship "
        "and the realistic odds for a strong-but-non-elite candidate."
    )
    prompt = (
        "Estimate application complexity. Reply ONLY with JSON:\n"
        '{"estimated_hours": number, '
        '"difficulty": "low|medium|high", '
        '"odds": "long_shot|moderate|good", '
        '"notes": "what drives the effort (essays, references, etc.)"}\n\n'
        f"TEXT:\n{_trim(text)}"
    )
    res = call_model("estimate_complexity", prompt, system=system, max_tokens=400)
    return extract_json(res["content"]) or {
        "estimated_hours": None, "difficulty": "unknown",
        "odds": "unknown", "notes": "estimation failed",
    }


# ── 10. score_opportunity (CLAUDE) ──────────────────────────────────────────
def score_opportunity(data: dict, profile: dict) -> dict:
    return _score_opportunity(data, profile)


# ── 11. check_known_scam (no model) ─────────────────────────────────────────
def check_known_scam(name: str, url: str = "") -> bool:
    """True if hard-blocked as a known scam (matches pipeline's boolean use)."""
    return _check_known_scam(name, url)["is_scam"]


def known_scam_detail(name: str, url: str = "") -> dict:
    return _check_known_scam(name, url)


# ── 12-13. persistence (no model) ───────────────────────────────────────────
def save_opportunity(data: dict) -> str:
    return db.save_opportunity(data)


def is_already_seen(url_or_name: str) -> bool:
    return db.is_already_seen(url_or_name)


# ── 14. send_notification (no model) — implemented in notifier ──────────────
def send_notification(report: str, channel: str = "all", subject: str = None):
    import notifier
    return notifier.send_notification(report, channel=channel, subject=subject)


# ── 15. add_to_calendar (no model) ──────────────────────────────────────────
def add_to_calendar(data: dict) -> dict:
    import calendar_sync
    return calendar_sync.add_to_calendar(data)


# ── 16. generate_cover_letter (GEMINI) ──────────────────────────────────────
def generate_cover_letter(data: dict, profile: dict) -> dict:
    return _generate_cover_letter(data, profile)


# ── 17-18. watchlist + tracker (no model) ───────────────────────────────────
def add_to_watchlist(data: dict) -> str:
    return db.add_to_watchlist(data)


def update_tracker(oid: str, status: str, note: str = "") -> bool:
    return db.update_tracker(oid, status, note=note)


# ── extra: email subject line (GEMINI) ──────────────────────────────────────
def generate_email_subject(num_matches: int, top_name: str = "") -> str:
    """Generate a punchy email subject. Falls back to a static line."""
    fallback = (
        f"🎯 OpportunityBot: {num_matches} match"
        f"{'es' if num_matches != 1 else ''} found"
        + (f" — top: {top_name}" if top_name else "")
    )
    if num_matches == 0:
        return "🎯 OpportunityBot: daily scan complete (no new matches)"
    try:
        prompt = (
            f"Write ONE short email subject line (max 70 chars) for a daily "
            f"alert about {num_matches} fellowship/scholarship matches. "
            f"Top match: {top_name}. Reply with the subject line only, no quotes."
        )
        res = call_model("generate_email_subject", prompt, max_tokens=40)
        line = (res["content"] or "").strip().splitlines()[0].strip().strip('"')
        return line[:120] or fallback
    except Exception:
        return fallback
