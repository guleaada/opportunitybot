"""
search.py — web search + URL fetching with caching.

- web_search()  : Google Custom Search JSON API → list[SearchResult].
- fetch_url()   : HTTP GET + BeautifulSoup text extraction, cached 24h on disk.

Neither function calls an LLM. ``clean_html`` (also model-free) lives in tools.py.
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

from rate_limiter import wait_for_quota, DailyQuotaExceeded

CACHE_DIR = Path(os.getenv("URL_CACHE_DIR", "data/url_cache"))
CACHE_TTL_SECONDS = 24 * 3600
# Plain browser UA. The old string appended "OpportunityBot/1.0", which
# self-identifies as a bot and is routinely 403'd by Cloudflare/Wordfence on
# the WordPress opportunity sites we fetch from.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


# Richer per-opportunity fields carried alongside the original four. All
# default to None so every existing call site keeps working unchanged.
OPPORTUNITY_FIELDS = (
    "category", "official_url", "location", "remote", "deadline", "reward",
    "estimated_value", "eligibility_status", "credibility_status",
    "source_quality", "requirements",
)


class SearchResult:
    def __init__(self, title: str, url: str, snippet: str = "", source: str = "",
                 category=None, official_url=None, location=None, remote=None,
                 deadline=None, reward=None, estimated_value=None,
                 eligibility_status=None, credibility_status=None,
                 source_quality=None, requirements=None):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        # Richer model — populated as the pipeline learns more; None until then.
        self.category = category
        self.official_url = official_url
        self.location = location
        self.remote = remote
        self.deadline = deadline
        self.reward = reward
        self.estimated_value = estimated_value
        self.eligibility_status = eligibility_status
        self.credibility_status = credibility_status
        self.source_quality = source_quality
        self.requirements = requirements if requirements is not None else []

    def to_dict(self) -> dict:
        base = {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
        }
        base.update({f: getattr(self, f) for f in OPPORTUNITY_FIELDS})
        return base

    @property
    def __dict__(self):  # convenience for {**result.__dict__}
        return self.to_dict()

    def __repr__(self):
        return f"<SearchResult {self.title!r} {self.url}>"


# ── Google Custom Search: rate limiting + 429 circuit breaker ──────────────
# The free CSE tier is 100 queries/day. Previously 73 queries fired back to
# back with no throttle and no breaker, so a quota exhaustion at query 3 still
# produced 70 more doomed requests. Backoff is bounded and gives up.
GOOGLE_MAX_CONSECUTIVE_429 = int(os.getenv("GOOGLE_MAX_CONSECUTIVE_429", "3"))
GOOGLE_BACKOFF_BASE_SECONDS = float(os.getenv("GOOGLE_BACKOFF_BASE_SECONDS", "2"))
GOOGLE_BACKOFF_MAX_SECONDS = float(os.getenv("GOOGLE_BACKOFF_MAX_SECONDS", "30"))

_google = {
    "attempted": 0, "successful": 0, "results": 0,
    "429": 0, "403": 0, "other_errors": 0,
    "consecutive_429": 0, "disabled": False, "disabled_reason": "",
}


def reset_google_state() -> None:
    """Call at the start of each scan so counters and the breaker are per-run."""
    _google.update({
        "attempted": 0, "successful": 0, "results": 0,
        "429": 0, "403": 0, "other_errors": 0,
        "consecutive_429": 0, "disabled": False, "disabled_reason": "",
    })


def google_stats() -> dict:
    return dict(_google)


def google_disabled() -> bool:
    return bool(_google["disabled"])


def _retry_after_seconds(resp, attempt: int) -> float:
    """Honour Retry-After when the server sends it; else exponential backoff."""
    header = None
    try:
        header = (resp.headers or {}).get("Retry-After")
    except Exception:
        header = None
    if header:
        try:
            return max(0.0, min(float(header), GOOGLE_BACKOFF_MAX_SECONDS))
        except (TypeError, ValueError):
            pass
    return min(GOOGLE_BACKOFF_BASE_SECONDS * (2 ** attempt),
               GOOGLE_BACKOFF_MAX_SECONDS)


def _google_error_reason(resp) -> str:
    """Pull Google's own explanation out of the error body when present."""
    try:
        err = (resp.json() or {}).get("error") or {}
        reason = ""
        for d in err.get("errors") or []:
            reason = d.get("reason") or reason
        return f"{reason or err.get('status', '')}: {err.get('message', '')}".strip(": ")
    except Exception:
        return ""


def web_search(query: str, max_results: int = 10) -> List[SearchResult]:
    """Google Custom Search → list of SearchResult. No model call.

    Rate limited, backs off on 429 (respecting Retry-After) and trips a
    circuit breaker after GOOGLE_MAX_CONSECUTIVE_429 so a quota problem cannot
    become hundreds of failed requests. Returns [] on any failure — never
    raises, so discovery degrades to the other providers.
    """
    api_key = os.getenv("GOOGLE_CSE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    if not api_key or not cse_id:
        print("⚠️  GOOGLE_CSE_API_KEY / GOOGLE_CSE_ID not set — skipping web search.")
        return []
    if _google["disabled"]:
        return []

    results: List[SearchResult] = []
    fetched = 0
    start = 1
    while fetched < max_results and start <= 91:
        num = min(10, max_results - fetched)

        try:
            wait_for_quota("google")     # rpm/rpd throttle
        except DailyQuotaExceeded:
            _google["disabled"] = True
            _google["disabled_reason"] = "local daily request budget reached"
            print("⛔ Google: local daily request budget reached — "
                  "stopping Google discovery for this scan.")
            break

        _google["attempted"] += 1
        try:
            resp = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={"key": api_key, "cx": cse_id, "q": query,
                        "num": num, "start": start},
                timeout=20,
            )
        except requests.RequestException as e:
            _google["other_errors"] += 1
            print(f"⚠️  Google search failed for {query!r}: {e}")
            break

        status = resp.status_code
        if status == 429:
            _google["429"] += 1
            _google["consecutive_429"] += 1
            n = _google["consecutive_429"]
            reason = _google_error_reason(resp)
            if n >= GOOGLE_MAX_CONSECUTIVE_429:
                _google["disabled"] = True
                _google["disabled_reason"] = reason or "repeated 429"
                print(f"⛔ Google 429 #{n} ({reason or 'rate/quota limit'}) — "
                      f"stopping Google discovery for this scan.")
                break
            delay = _retry_after_seconds(resp, n - 1)
            print(f"⚠️  Google 429 #{n} ({reason or 'rate/quota limit'}) — "
                  f"backing off {delay:.1f}s")
            time.sleep(delay)
            continue                      # retry this page, bounded by the breaker

        if status == 403:
            _google["403"] += 1
            _google["disabled"] = True
            _google["disabled_reason"] = _google_error_reason(resp) or "403 forbidden"
            print(f"⛔ Google 403 ({_google['disabled_reason']}) — "
                  f"stopping Google discovery for this scan.")
            break

        if status != 200:
            _google["other_errors"] += 1
            print(f"⚠️  Google HTTP {status} for {query!r}: "
                  f"{_google_error_reason(resp)}")
            break

        try:
            data = resp.json()
        except ValueError as e:
            _google["other_errors"] += 1
            print(f"⚠️  Google returned unparseable JSON for {query!r}: {e}")
            break

        _google["successful"] += 1
        _google["consecutive_429"] = 0    # a success clears the streak

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

    _google["results"] += len(results)
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


# ── Reader-proxy fetch (r.jina.ai) ─────────────────────────────────────────
# The runner's own IP is Cloudflare-blocked on most opportunity sites. Jina
# fetches server-side and returns clean article text, so we get the FULL page
# instead of falling all the way back to a short feed snippet.
JINA_READER_PREFIX = "https://r.jina.ai/"
JINA_TIMEOUT = 30
_JINA_MIN_INTERVAL = 0.5   # be polite; keyless free tier is ~20 req/min
_last_jina_call = 0.0


def _host_of(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url if "//" in url else "//" + url).netloc or url
    except Exception:
        return url or "?"


def fetch_via_jina(url: str) -> dict:
    """Fetch a page's full text through the r.jina.ai reader proxy.

    Returns the SAME dict shape as ``fetch_url`` so callers need no changes.
    Never raises — any failure comes back as an error dict so the caller can
    fall through to its next tier.
    """
    global _last_jina_call
    host = _host_of(url)
    if not url:
        return {"url": url, "html": "", "text": "", "status": 0,
                "cached": False, "error": "empty url"}
    try:
        gap = time.time() - _last_jina_call
        if gap < _JINA_MIN_INTERVAL:
            time.sleep(_JINA_MIN_INTERVAL - gap)

        headers = {"User-Agent": USER_AGENT, "Accept": "text/plain"}
        # Keyless works; a free key just raises the rate limit.
        api_key = os.getenv("JINA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        print(f"↩️  jina reader for {host}")
        resp = requests.get(JINA_READER_PREFIX + url, headers=headers,
                            timeout=JINA_TIMEOUT)
        _last_jina_call = time.time()

        status = resp.status_code
        text = resp.text if status == 200 else ""
        if status != 200 or not (text or "").strip():
            print(f"⚠️  jina failed for {host}: status {status}")
            return {"url": url, "html": "", "text": "", "status": status,
                    "cached": False, "error": f"jina status {status}"}
        return {"url": url, "html": "", "text": text, "status": status,
                "cached": False, "error": None}
    except Exception as e:
        _last_jina_call = time.time()
        print(f"⚠️  jina failed for {host}: {e}")
        return {"url": url, "html": "", "text": "", "status": 0,
                "cached": False, "error": f"{type(e).__name__}: {e}"}


def fetch_url(url: str, force: bool = False) -> dict:
    """Fetch a URL (cached 24h). No model call.

    Returns ``{"url", "html", "text", "status", "cached", "error"}``.
    ``text`` is a local BeautifulSoup extraction; pass it through
    ``tools.clean_html`` for further local normalization when needed.
    """
    if not force:
        try:
            cached = _read_cache(url)
        except Exception:  # a corrupt cache entry must not kill the fetch
            cached = None
        if cached is not None:
            return {**cached, "cached": True, "error": None}

    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=25)
        status = resp.status_code
        html = resp.text if status == 200 else ""
        text = _basic_text(html) if html else ""
        if status == 200:
            try:
                _write_cache(url, html, text, status)
            except OSError as e:
                # Disk full / read-only FS must not lose a good fetch.
                print(f"⚠️  Could not cache {url}: {e}")
        return {"url": url, "html": html, "text": text, "status": status,
                "cached": False, "error": None}
    except requests.RequestException as e:
        return {"url": url, "html": "", "text": "", "status": 0,
                "cached": False, "error": str(e)}
    except Exception as e:
        # Anything else (HTML parser failure, decoding error, ...). fetch_url
        # must ALWAYS return its dict — a raise here escapes before any counter
        # moves and silently kills the candidate.
        return {"url": url, "html": "", "text": "", "status": 0,
                "cached": False, "error": f"{type(e).__name__}: {e}"}
