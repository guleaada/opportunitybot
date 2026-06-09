"""
ci_persistence.py — bridges local vs GitHub Actions environments.

Locally: JSON files persist naturally on disk.
In GitHub Actions: runners are ephemeral, so the workflow commits the
``data/*.json`` files back to the repo after each run. That makes the bot
stateful across runs. This module helps the code behave identically in both.

NOTE ON DEFAULTS (deviation from the spec snippet, by design):
The spec snippet initialized seen_opportunities/watchlist as ``[]``. The actual
data layer (database.py) stores those as DICTS keyed by a stable opportunity id
(O(1) dedupe + upsert), and cost/activity logs as LISTS. Initializing the wrong
container would break is_already_seen(). So the defaults below match the real
data layer: dict for seen/tracker/watchlist, list for cost/activity.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

# Container shape MUST match how database.py / cost_tracker.py read each file.
_DEFAULTS = {
    "seen_opportunities.json": {},   # dict keyed by opportunity id
    "application_tracker.json": {},  # dict keyed by opportunity id
    "watchlist.json": {},            # dict keyed by opportunity id
    "cost_log.json": [],             # append-only list
    "activity_log.json": [],         # append-only list
}


def is_running_in_ci() -> bool:
    """True when running inside GitHub Actions (or any CI that sets CI=true)."""
    return os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"


def ensure_data_dir() -> None:
    """Create data/ and initialize any missing JSON files with correct shape."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "url_cache").mkdir(exist_ok=True)
    for filename, default_value in _DEFAULTS.items():
        path = DATA_DIR / filename
        if not path.exists():
            path.write_text(json.dumps(default_value, indent=2))
            print(f"📁 Initialized empty {filename}")


def log_run_metadata(mode: str, stats: dict) -> None:
    """Append a run record to activity_log.json for debugging/auditing.

    Coexists with database.log_activity() entries (both are dicts in the same
    list); this one carries a 'kind': 'run' marker so runs are filterable.
    """
    log_path = DATA_DIR / "activity_log.json"
    try:
        log = json.loads(log_path.read_text()) if log_path.exists() else []
        if not isinstance(log, list):
            log = []
    except (json.JSONDecodeError, OSError):
        log = []

    log.append({
        "at": datetime.now(timezone.utc).isoformat(),
        "kind": "run",
        "mode": mode,
        "environment": "github_actions" if is_running_in_ci() else "local",
        "stats": stats,
    })
    # Keep the log bounded.
    log = log[-200:]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))


def persistence_files():
    """Return the list of data files the CI workflow should commit back."""
    return [str(DATA_DIR / name) for name in _DEFAULTS]
