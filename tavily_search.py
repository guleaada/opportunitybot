"""
tavily_search.py — Tavily Search API → list[SearchResult].

Google's Custom Search JSON API is unavailable to this project (403
PERMISSION_DENIED: "This project does not have the access to Custom Search
JSON API"), so this is the replacement web-search backend.

Deliberately a drop-in for ``search.web_search``: same
``(query, max_results) -> List[SearchResult]`` signature, same
never-raise contract, same module-level per-scan counters and circuit
breaker. Nothing here is wired into discovery yet — provider selection is a
separate change.

API contract confirmed against Tavily's official Python client
(github.com/tavily-ai/tavily-python, tavily/tavily.py):

    POST https://api.tavily.com/search
    Authorization: Bearer <TAVILY_API_KEY>          (header, never a body field)
    body: {"query": ..., "max_results": ..., ...}   (None-valued keys omitted)
    200 -> {"query", "answer", "images", "results": [
                {"title", "url", "content", "score", "raw_content"} ],
            "response_time"}
    400 BadRequest · 401 InvalidAPIKey · 429 UsageLimitExceeded ·
    403/432/433 Forbidden (432/433 are plan/credit limits)
"""

import os
import time
from typing import List
from urllib.parse import urlparse

import requests

# The one result model for the whole project — never a second one.
from search import SearchResult
from rate_limiter import (wait_for_quota, register_provider,
                          DailyQuotaExceeded)

TAVILY_URL = "https://api.tavily.com/search"
TAVILY_TIMEOUT_SECONDS = float(os.getenv("TAVILY_TIMEOUT_SECONDS", "20"))

# Tavily returns everything in a single response — there is no offset/page
# parameter in the documented body — so one query is one request. This is the
# per-request ceiling the API documents.
TAVILY_MAX_RESULTS_PER_CALL = int(os.getenv("TAVILY_MAX_RESULTS_PER_CALL", "20"))

# Same shape as search.py's Google breaker: back off on 429, respect
# Retry-After, and give up after N in a row so a quota problem cannot become
# hundreds of doomed requests. Constants are Tavily's own — the Google ones
# are not reused because their tuning is not this provider's.
TAVILY_MAX_CONSECUTIVE_429 = int(os.getenv("TAVILY_MAX_CONSECUTIVE_429", "3"))
TAVILY_BACKOFF_BASE_SECONDS = float(os.getenv("TAVILY_BACKOFF_BASE_SECONDS", "2"))
TAVILY_BACKOFF_MAX_SECONDS = float(os.getenv("TAVILY_BACKOFF_MAX_SECONDS", "30"))

# Tavily maps all of these to "forbidden": plan limit reached / out of
# credits. Retrying inside one scan cannot help, so they disable the provider.
_FORBIDDEN_STATUSES = (403, 432, 433)

# Conservative, environment-overridable throttle. These are NOT Tavily's
# published free-tier limits — Tavily meters by credits, not by requests per
# minute or day, and does not document an RPM/RPD ceiling. They are a
# deliberately cautious local budget, in the same spirit as the Mistral
# placeholders, so a runaway scan cannot burn an account's credits. Raise them
# once real limits are known.
TAVILY_REQUESTS_PER_MINUTE = int(os.getenv("TAVILY_REQUESTS_PER_MINUTE", "10"))
TAVILY_REQUESTS_PER_DAY = int(os.getenv("TAVILY_REQUESTS_PER_DAY", "100"))

# Registering here (rather than hand-editing rate_limiter.LIMITS) creates the
# quota deques alongside the limits — the parallel-dicts pattern that caused
# KeyError('openrouter') is exactly what register_provider exists to prevent.
register_provider("tavily", TAVILY_REQUESTS_PER_MINUTE, TAVILY_REQUESTS_PER_DAY)

_BLANK_STATE = {
    "attempted": 0, "successful": 0, "results": 0,
    "429": 0, "403": 0, "other_errors": 0,
    "consecutive_429": 0, "disabled": False, "disabled_reason": "",
}

_state = dict(_BLANK_STATE)


def reset_state() -> None:
    """Call at the start of each scan so counters and the breaker are per-run."""
    _state.update(_BLANK_STATE)


def stats() -> dict:
    return dict(_state)


def disabled() -> bool:
    return bool(_state["disabled"])


def _api_key() -> str:
    return (os.getenv("TAVILY_API_KEY") or "").strip()


def _scrub(text) -> str:
    """Never let the key reach a log line, however it got into the message."""
    out = str(text)
    key = _api_key()
    if key and key in out:
        out = out.replace(key, "<redacted>")
    return out


def _retry_after_seconds(resp, attempt: int) -> float:
    """Honour Retry-After when the server sends it; else exponential backoff."""
    header = None
    try:
        header = (resp.headers or {}).get("Retry-After")
    except Exception:
        header = None
    if header:
        try:
            return max(0.0, min(float(header), TAVILY_BACKOFF_MAX_SECONDS))
        except (TypeError, ValueError):
            pass
    return min(TAVILY_BACKOFF_BASE_SECONDS * (2 ** attempt),
               TAVILY_BACKOFF_MAX_SECONDS)


def _error_reason(resp) -> str:
    """Tavily's own explanation, when the body carries one. Never the key."""
    try:
        body = resp.json()
    except Exception:
        return ""
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return _scrub(err.get("message") or err.get("code") or "")
        if err:
            return _scrub(err)
        if body.get("detail"):
            return _scrub(body["detail"])
    return ""


def _source_of(url: str) -> str:
    """Hostname, as the discovery/prioritizer layers expect for a search hit.

    Left exactly as the URL spells it apart from case — the feed-domain sets
    the prioritizer matches against include the "www." form, so stripping it
    would silently reclassify results.
    """
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def test_tavily() -> dict:
    """One minimal live Tavily request, for `main.py --test`.

    Reuses tavily_search() rather than building a second client, so the probe
    exercises the same endpoint, auth, parsing and error handling the scan
    uses. Returns {"ok": True, "provider": "tavily", "results": n} or
    {"ok": False, "error": ...}.

    Per-scan state is snapshotted and restored, so probing can never leave
    the breaker tripped or the counters dirty — the same property the model
    provider probes have.
    """
    if not _api_key():
        return {"ok": False, "error": "TAVILY_API_KEY not set"}

    before = dict(_state)
    try:
        reset_state()
        results = tavily_search("test", max_results=1)
        s = stats()
        if s["successful"]:
            return {"ok": True, "provider": "tavily", "results": len(results)}
        # tavily_search never raises, so the reason lives in the state.
        error = s.get("disabled_reason") or ""
        if not error:
            if s["other_errors"]:
                error = f"{s['other_errors']} request error(s) — see log above"
            elif s["429"]:
                error = "rate limited"
            else:
                error = "no successful response"
        return {"ok": False, "error": error}
    except Exception as e:                    # a probe must never crash --test
        return {"ok": False, "error": _scrub(e)}
    finally:
        _state.clear()
        _state.update(before)


def tavily_search(query: str, max_results: int = 10) -> List[SearchResult]:
    """Tavily Search → list of SearchResult. No model call.

    Returns [] on any failure and never raises, so discovery degrades to the
    other providers exactly as it does when Google fails.
    """
    api_key = _api_key()
    if not api_key:
        print("⚠️  TAVILY_API_KEY not set — skipping web search.")
        return []
    if _state["disabled"]:
        return []
    if not (query or "").strip():
        return []

    want = max(1, min(int(max_results or 1), TAVILY_MAX_RESULTS_PER_CALL))
    payload = {"query": query, "max_results": want,
               "search_depth": "basic", "include_answer": False,
               "include_raw_content": False, "include_images": False}
    headers = {"Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}

    attempt = 0
    while True:
        try:
            wait_for_quota("tavily")          # local rpm/rpd throttle
        except DailyQuotaExceeded:
            _state["disabled"] = True
            _state["disabled_reason"] = "local daily request budget reached"
            print("⛔ Tavily: local daily request budget reached — "
                  "stopping Tavily discovery for this scan.")
            return []

        _state["attempted"] += 1
        try:
            resp = requests.post(TAVILY_URL, json=payload, headers=headers,
                                 timeout=TAVILY_TIMEOUT_SECONDS)
        except requests.RequestException as e:
            _state["other_errors"] += 1
            print(f"⚠️  Tavily search failed for {query!r}: {_scrub(e)}")
            return []

        status = getattr(resp, "status_code", 0)

        if status == 429:
            _state["429"] += 1
            _state["consecutive_429"] += 1
            n = _state["consecutive_429"]
            reason = _error_reason(resp)
            if n >= TAVILY_MAX_CONSECUTIVE_429:
                _state["disabled"] = True
                _state["disabled_reason"] = reason or "repeated 429"
                print(f"⛔ Tavily 429 #{n} ({reason or 'usage limit'}) — "
                      f"stopping Tavily discovery for this scan.")
                return []
            delay = _retry_after_seconds(resp, attempt)
            print(f"⚠️  Tavily 429 #{n} ({reason or 'usage limit'}) — "
                  f"backing off {delay:.1f}s")
            time.sleep(delay)
            attempt += 1
            continue                      # retry, bounded by the breaker

        if status in _FORBIDDEN_STATUSES:
            _state["403"] += 1
            _state["disabled"] = True
            _state["disabled_reason"] = (
                _error_reason(resp) or f"{status} forbidden / plan limit")
            print(f"⛔ Tavily {status} ({_state['disabled_reason']}) — "
                  f"stopping Tavily discovery for this scan.")
            return []

        if status == 401:
            # A rejected key cannot fix itself mid-scan. Name the variable,
            # never the value.
            _state["other_errors"] += 1
            _state["disabled"] = True
            _state["disabled_reason"] = (
                _error_reason(resp) or "401 invalid TAVILY_API_KEY")
            print(f"⛔ Tavily 401 — TAVILY_API_KEY was rejected "
                  f"({_state['disabled_reason']}); stopping Tavily discovery "
                  f"for this scan.")
            return []

        if status != 200:
            _state["other_errors"] += 1
            print(f"⚠️  Tavily HTTP {status} for {query!r}: {_error_reason(resp)}")
            return []

        break

    try:
        data = resp.json()
    except Exception as e:
        _state["other_errors"] += 1
        print(f"⚠️  Tavily returned unparseable JSON for {query!r}: {_scrub(e)}")
        return []
    if not isinstance(data, dict):
        _state["other_errors"] += 1
        print(f"⚠️  Tavily returned an unexpected payload for {query!r}.")
        return []

    _state["successful"] += 1
    _state["consecutive_429"] = 0         # a success clears the streak

    results: List[SearchResult] = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        if not url:
            continue                      # a result with no URL is unusable
        results.append(SearchResult(
            title=(item.get("title") or "").strip(),
            url=url,
            snippet=(item.get("content") or "").strip(),
            source=_source_of(url),
        ))

    results = results[:want]
    _state["results"] += len(results)
    return results
