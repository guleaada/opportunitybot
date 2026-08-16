"""
api_sources.py — key-less JSON opportunity APIs for the APIProvider.

Deliberately SMALL. Two sources, both chosen because they target the gap in
the current mix (the live RSS feeds are student-scholarship aggregators, while
the profile is a working AI professional):

  remotive  — remote software/dev roles. Free, no auth, documented JSON.
              https://remotive.com/api/remote-jobs?category=software-dev
              Publisher asks callers to cache and not poll aggressively; one
              call per scan is well inside that.

  arbeitnow — remote/EU job board, many dev roles. Free, no auth, JSON,
              paginated. https://www.arbeitnow.com/api/job-board-api

Each adapter normalizes to the shape APIProvider expects:
    [{"title", "url", "description", "category"}]

Every adapter is independent and fail-soft: a network error, an HTML error
page, a schema change or a rate limit returns [] and logs, so one bad API can
never stop discovery.

NOT included on purpose: sources needing paid keys, sources with no public
API (Devpost), and sources whose results would be mostly ineligible.
"""

import json
import os
import urllib.error
import urllib.request

TIMEOUT = int(os.getenv("API_SOURCE_TIMEOUT", "20"))
MAX_ITEMS_PER_SOURCE = int(os.getenv("API_MAX_ITEMS_PER_SOURCE", "40"))
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")


def _get_json(url: str):
    """GET → parsed JSON. Raises on failure; callers convert to []."""
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8", errors="replace"))


def _clean(text, limit=2000) -> str:
    import re
    t = re.sub(r"<[^>]+>", " ", str(text or ""))
    return re.sub(r"\s+", " ", t).strip()[:limit]


# ── Adapters ───────────────────────────────────────────────────────────────
def remotive(url: str = None) -> list:
    """Remotive remote-jobs API → normalized candidates."""
    url = url or os.getenv(
        "REMOTIVE_API_URL",
        "https://remotive.com/api/remote-jobs?category=software-dev")
    data = _get_json(url)
    jobs = (data or {}).get("jobs") or []
    out = []
    for j in jobs[:MAX_ITEMS_PER_SOURCE]:
        title = (j.get("title") or "").strip()
        link = (j.get("url") or "").strip()
        if not title or not link:
            continue
        bits = [
            _clean(j.get("description"), 1500),
            f"Company: {j.get('company_name')}" if j.get("company_name") else "",
            f"Location: {j.get('candidate_required_location')}"
            if j.get("candidate_required_location") else "",
            f"Salary: {j.get('salary')}" if j.get("salary") else "",
            f"Type: {j.get('job_type')}" if j.get("job_type") else "",
            f"Published: {j.get('publication_date')}"
            if j.get("publication_date") else "",
        ]
        out.append({"title": title, "url": link,
                    "description": " | ".join(b for b in bits if b),
                    "category": "remote_jobs"})
    return out


def arbeitnow(url: str = None) -> list:
    """Arbeitnow job-board API → normalized candidates."""
    url = url or os.getenv(
        "ARBEITNOW_API_URL", "https://www.arbeitnow.com/api/job-board-api")
    data = _get_json(url)
    jobs = (data or {}).get("data") or []
    out = []
    for j in jobs[:MAX_ITEMS_PER_SOURCE]:
        title = (j.get("title") or "").strip()
        link = (j.get("url") or "").strip()
        if not title or not link:
            continue
        bits = [
            _clean(j.get("description"), 1500),
            f"Company: {j.get('company_name')}" if j.get("company_name") else "",
            f"Location: {j.get('location')}" if j.get("location") else "",
            "Remote: yes" if j.get("remote") else "",
            f"Tags: {', '.join(j.get('tags') or [])}" if j.get("tags") else "",
        ]
        out.append({"title": title, "url": link,
                    "description": " | ".join(b for b in bits if b),
                    "category": "remote_jobs" if j.get("remote") else "dev_jobs"})
    return out


# name -> (adapter, category, why it is here)
REGISTRY = {
    "remotive": (remotive, "remote_jobs",
                 "free key-less remote software-dev jobs"),
    "arbeitnow": (arbeitnow, "dev_jobs",
                  "free key-less job board, many dev roles"),
}

# Enabled by default; set API_SOURCES to a comma-separated subset (or "" to
# disable all) without touching code.
def enabled_sources() -> list:
    raw = os.getenv("API_SOURCES")
    if raw is None:
        return list(REGISTRY)
    return [n.strip() for n in raw.split(",") if n.strip() in REGISTRY]


def fetch(name: str) -> list:
    """Fetch one named source. Never raises — returns [] and logs on failure."""
    entry = REGISTRY.get(name)
    if not entry:
        return []
    adapter = entry[0]
    try:
        items = adapter() or []
        print(f"🛰️  API {name}: +{len(items)} candidates")
        return items
    except urllib.error.HTTPError as e:
        print(f"⚠️  API {name} failed: HTTP {e.code}")
    except urllib.error.URLError as e:
        print(f"⚠️  API {name} failed: {e.reason}")
    except (ValueError, TypeError) as e:
        print(f"⚠️  API {name} returned unusable data: {e}")
    except Exception as e:
        print(f"⚠️  API {name} failed: {type(e).__name__}: {e}")
    return []
