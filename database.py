"""
database.py — JSON-file memory / history layer.

All persistent state lives as plain JSON under ``data/`` so it's trivially
inspectable and needs no external DB. Thread-safe per-file via a lock.

Files:
  seen_opportunities.json   — every opportunity ever processed (dedupe + audit)
  application_tracker.json  — mini-CRM of things being applied to
  watchlist.json            — not-yet-open opportunities to recheck
  activity_log.json         — append-only event/error log
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
SEEN_PATH = DATA_DIR / "seen_opportunities.json"
TRACKER_PATH = DATA_DIR / "application_tracker.json"
WATCHLIST_PATH = DATA_DIR / "watchlist.json"
ACTIVITY_PATH = DATA_DIR / "activity_log.json"

_locks = {}


def _lock_for(path: Path) -> Lock:
    key = str(path)
    if key not in _locks:
        _locks[key] = Lock()
    return _locks[key]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text() or json.dumps(default))
    except (json.JSONDecodeError, OSError):
        return default


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def opportunity_id(url: str = "", name: str = "") -> str:
    """Stable id from URL (preferred) or name."""
    basis = (url or name or "").strip().lower().rstrip("/")
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


# ── seen_opportunities ─────────────────────────────────────────────────────
def _load_seen() -> dict:
    return _read(SEEN_PATH, {})


def is_already_seen(url_or_name: str) -> bool:
    """True if we've already processed this URL/name. No model call."""
    seen = _load_seen()
    oid = opportunity_id(url=url_or_name, name=url_or_name)
    if oid in seen:
        return True
    # Also match by raw url field for safety.
    low = (url_or_name or "").strip().lower().rstrip("/")
    return any((v.get("url", "").strip().lower().rstrip("/") == low) for v in seen.values())


def save_opportunity(data: dict) -> str:
    """Upsert an opportunity record into seen_opportunities.json.

    ``data`` should contain at least ``url`` and/or ``title``/``name`` and a
    ``status`` (e.g. filtered_out, scam, ineligible, analyzed). Returns the id.
    """
    with _lock_for(SEEN_PATH):
        seen = _load_seen()
        url = data.get("url", "")
        name = data.get("title") or data.get("name") or ""
        oid = data.get("id") or opportunity_id(url=url, name=name)
        existing = seen.get(oid, {})
        record = {**existing, **data, "id": oid}
        record.setdefault("first_seen", _now())
        record["last_updated"] = _now()
        seen[oid] = record
        _write(SEEN_PATH, seen)
    return oid


def get_opportunity(oid: str) -> dict:
    return _load_seen().get(oid, {})


def all_seen() -> dict:
    return _load_seen()


# ── application_tracker (mini CRM) ──────────────────────────────────────────
VALID_STATUSES = [
    "interested", "preparing", "documents_ready", "submitted",
    "under_review", "interview", "accepted", "rejected", "withdrawn",
]


def _load_tracker() -> dict:
    return _read(TRACKER_PATH, {})


def add_to_tracker(data: dict) -> str:
    with _lock_for(TRACKER_PATH):
        tracker = _load_tracker()
        oid = data.get("id") or opportunity_id(
            url=data.get("url", ""), name=data.get("title") or data.get("name", ""))
        tracker[oid] = {
            **tracker.get(oid, {}),
            **data,
            "id": oid,
            "status": data.get("status", "interested"),
            "updated": _now(),
        }
        tracker[oid].setdefault("added", _now())
        _write(TRACKER_PATH, tracker)
    return oid


def update_tracker(oid: str, status: str, note: str = "") -> bool:
    with _lock_for(TRACKER_PATH):
        tracker = _load_tracker()
        if oid not in tracker:
            return False
        tracker[oid]["status"] = status
        tracker[oid]["updated"] = _now()
        if note:
            tracker[oid].setdefault("notes", []).append(
                {"at": _now(), "note": note})
        _write(TRACKER_PATH, tracker)
    return True


def all_tracked() -> dict:
    return _load_tracker()


# ── watchlist ───────────────────────────────────────────────────────────────
def _load_watchlist() -> dict:
    return _read(WATCHLIST_PATH, {})


def add_to_watchlist(data: dict) -> str:
    with _lock_for(WATCHLIST_PATH):
        wl = _load_watchlist()
        oid = data.get("id") or opportunity_id(
            url=data.get("url", ""), name=data.get("title") or data.get("name", ""))
        wl[oid] = {**wl.get(oid, {}), **data, "id": oid, "added": _now()}
        _write(WATCHLIST_PATH, wl)
    return oid


def remove_from_watchlist(oid: str) -> bool:
    with _lock_for(WATCHLIST_PATH):
        wl = _load_watchlist()
        if oid in wl:
            del wl[oid]
            _write(WATCHLIST_PATH, wl)
            return True
    return False


def all_watchlist() -> dict:
    return _load_watchlist()


# ── activity log ─────────────────────────────────────────────────────────────
def log_activity(kind: str, message: str, **extra) -> None:
    with _lock_for(ACTIVITY_PATH):
        log = _read(ACTIVITY_PATH, [])
        log.append({"at": _now(), "kind": kind, "message": message, **extra})
        # Keep the log bounded.
        if len(log) > 5000:
            log = log[-5000:]
        _write(ACTIVITY_PATH, log)


def log_error(message: str, **extra) -> None:
    print(f"❌ {message}")
    log_activity("error", message, **extra)
