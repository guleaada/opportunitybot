"""
search.py — web search + URL fetching with caching.

- web_search()  : Google Custom Search JSON API → list[SearchResult].
- fetch_url()   : HTTP GET + BeautifulSoup text extraction, cached 24h on disk.

Neither function calls an LLM. ``clean_html`` (Groq) lives in tools.py.
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

CACHE_DIR = Path(os.getenv("URL_CACHE_DIR", "data/url_cache"))
CACHE_TTL_SECONDS = 24 * 3600
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 OpportunityBot/1.0"
)

RSS_FEEDS = [
    "https://opportunitiescorners.com/feed/",
    "https://www.opportunitiesforafricans.com/feed/",
    "https://opportunitydesk.org/feed/",
    "https://www.youthop.com/feed",
]


class SearchResult:
    def __init__(self, title: str, url: str, snippet: str = "", source: str = ""):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
        }

    @property
    def __dict__(self):  # convenience for {**result.__dict__}
        return self.to_dict()

    def __repr__(self):
        return f"<SearchResult {self.title!r} {self.url}>"


def web_search(query: str, max_results: int = 10) -> List[SearchResult]:
    """Google Custom Search → list of SearchResult. No model call.

    Returns an empty list (and prints a warning) if CSE keys are missing, so
    the rest of the pipeline degrades gracefully.
    """
    api_key = os.getenv("GOOGLE_CSE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        print("⚠️  GOOGLE_CSE_API_KEY / GOOGLE_CSE_ID not set — skipping web search.")
        return []

    results: List[SearchResult] = []
    # CSE returns up to 10 per page; paginate via `start`.
    fetched = 0
    start = 1
    while fetched < max_results and start <= 91:
        num = min(10, max_results - fetched)
        try:
            resp = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": api_key,
                    "cx": cse_id,
                    "q": query,
                    "num": num,
                    "start": start,
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            print(f"⚠️  Search failed for {query!r}: {e}")
            break

        items = data.get("items", [])
        if not items:
            break
        for it in items:
            results.append(SearchResult(
                title=it.get("title", ""),
                url=it.get("link", ""),
                snippet=it.get("snippet", ""),
                source=it.get("displayLink", ""),
            ))
        fetched += len(items)
        start += len(items)
        if len(items) < num:
            break

    return results[:max_results]


# ── URL fetch with 24h disk cache ──────────────────────────────────────────
def _cache_path(url: str) -> Path:
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.json"


def _read_cache(url: str) -> Optional[dict]:
    path = _cache_path(url)
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - blob.get("fetched_at", 0) > CACHE_TTL_SECONDS:
        return None
    return blob


def _write_cache(url: str, html: str, text: str, status: int) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(url).write_text(json.dumps({
        "url": url,
        "fetched_at": time.time(),
        "fetched_iso": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "html": html,
        "text": text,
    }, ensure_ascii=False))


def _basic_text(html: str) -> str:
    """Local HTML→text via BeautifulSoup (no model). Groq cleanup is optional."""
    soup = BeautifulSoup(html, "lxml") if _has_lxml() else BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _has_lxml() -> bool:
    try:
        import lxml  # noqa: F401
        return True
    except ImportError:
        return False


def fetch_url(url: str, force: bool = False) -> dict:
    """Fetch a URL (cached 24h). No model call.

    Returns ``{"url", "html", "text", "status", "cached", "error"}``.
    ``text`` is a local BeautifulSoup extraction; pass it through
    ``tools.clean_html`` for Groq-polished text when needed.
    """
    if not force:
        cached = _read_cache(url)
        if cached is not None:
            return {**cached, "cached": True, "error": None}

    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=25)
        status = resp.status_code
        html = resp.text if status == 200 else ""
        text = _basic_text(html) if html else ""
        if status == 200:
            _write_cache(url, html, text, status)
        return {"url": url, "html": html, "text": text, "status": status,
                "cached": False, "error": None}
    except requests.RequestException as e:
        return {"url": url, "html": "", "text": "", "status": 0,
                "cached": False, "error": str(e)}


def _parse_rss_entries(feed_url: str, max_items: int) -> list:
    """Fetch and parse an RSS/Atom feed. Tries feedparser first (preferred in
    production); falls back to stdlib requests+ElementTree if feedparser is
    unavailable (e.g. sgmllib3k build fails in some environments).
    Returns a list of dicts with keys: title, link, summary.
    """
    import xml.etree.ElementTree as ET  # noqa: PLC0415
    from urllib.parse import urlparse as _up  # noqa: PLC0415

    ua = (
        "Mozilla/5.0 (compatible; OpportunityBot/1.0; "
        "+https://github.com/guleaada/opportunitybot)"
    )

    try:
        import feedparser  # noqa: PLC0415
        feed = feedparser.parse(feed_url, agent=ua)
        return [
            {
                "title": getattr(e, "title", ""),
                "link": getattr(e, "link", ""),
                "summary": getattr(e, "summary", ""),
            }
            for e in feed.entries[:max_items]
        ]
    except ImportError:
        pass  # feedparser not available — use stdlib fallback below

    resp = requests.get(feed_url, headers={"User-Agent": ua}, timeout=20)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    ATOM = "http://www.w3.org/2005/Atom"
    raw_items = root.findall(".//item") or root.findall(f".//{{{ATOM}}}entry")

    entries = []
    for item in raw_items[:max_items]:
        def _text(tag, atom_tag=None):
            el = item.find(tag)
            if el is None and atom_tag:
                el = item.find(atom_tag)
            if el is None:
                return ""
            return (el.text or el.get("href", "") or "").strip()

        entries.append({
            "title": _text("title", f"{{{ATOM}}}title"),
            "link": _text("link", f"{{{ATOM}}}link"),
            "summary": _text("description", f"{{{ATOM}}}summary"),
        })
    return entries


def fetch_rss_feeds(
    feeds: List[str] = None, max_items_per_feed: int = 20
) -> List[SearchResult]:
    """Pull fresh posts from multiple WordPress RSS feeds.

    Uses a browser-like User-Agent to bypass Cloudflare bot checks.
    Per-feed errors are caught and logged; one bad feed never stops the others.
    Returns the combined list across all feeds.
    """
    from urllib.parse import urlparse  # noqa: PLC0415

    if feeds is None:
        feeds = RSS_FEEDS

    all_results: List[SearchResult] = []
    feeds_ok = 0

    for feed_url in feeds:
        domain = urlparse(feed_url).netloc or feed_url
        try:
            count_before = len(all_results)
            for entry in _parse_rss_entries(feed_url, max_items_per_feed):
                link = entry.get("link", "").strip()
                title = entry.get("title", "").strip()
                if not link or not title:
                    continue
                raw_summary = entry.get("summary", "")
                snippet = _basic_text(raw_summary) if raw_summary else ""
                all_results.append(SearchResult(
                    title=title,
                    url=link,
                    snippet=snippet,
                    source=domain,
                ))
            feeds_ok += 1
            print(f"✅ RSS {domain}: {len(all_results) - count_before} posts")
        except Exception as e:
            print(f"⚠️ RSS feed {domain} failed: {e}")

    print(
        f"📰 RSS total: {len(all_results)} posts from {feeds_ok}/{len(feeds)} feeds"
    )
    return all_results
