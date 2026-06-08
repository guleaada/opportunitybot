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
