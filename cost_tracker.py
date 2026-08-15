"""
cost_tracker.py — track API spend per provider.

Critical for protecting a small Claude credit. Every model call writes one
entry to ``data/cost_log.json``. Helpers expose daily/monthly rollups that the
router and the reports read.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

COST_LOG_PATH = Path(os.getenv("COST_LOG_PATH", "data/cost_log.json"))
_lock = Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_usage(usage):
    """Normalize anthropic/groq usage objects or dicts into (in, out) ints."""
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        tin = usage.get("input", usage.get("input_tokens", usage.get("prompt_tokens", 0)))
        tout = usage.get("output", usage.get("output_tokens", usage.get("completion_tokens", 0)))
        return int(tin or 0), int(tout or 0)
    tin = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
    tout = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
    return int(tin or 0), int(tout or 0)


def _read_log() -> list:
    if not COST_LOG_PATH.exists():
        return []
    try:
        return json.loads(COST_LOG_PATH.read_text() or "[]")
    except (json.JSONDecodeError, OSError):
        return []


def log_cost(provider: str, task: str, usage, cost_usd: float) -> None:
    """Append a single cost entry to the log."""
    tin, tout = _coerce_usage(usage)
    entry = {
        "timestamp": _now_iso(),
        "provider": provider,
        "task": task,
        "tokens_in": tin,
        "tokens_out": tout,
        "cost_usd": round(float(cost_usd), 6),
    }
    with _lock:
        COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log = _read_log()
        log.append(entry)
        COST_LOG_PATH.write_text(json.dumps(log, indent=2))


def get_daily_claude_spend(date_str: str = None) -> float:
    """Return Claude spend (USD) for the given UTC date (today by default)."""
    today = date_str or datetime.now(timezone.utc).date().isoformat()
    return sum(
        e["cost_usd"]
        for e in _read_log()
        if e["provider"] == "claude" and e["timestamp"].startswith(today)
    )


def get_monthly_claude_spend(month_str: str = None) -> float:
    month = month_str or datetime.now(timezone.utc).strftime("%Y-%m")
    return sum(
        e["cost_usd"]
        for e in _read_log()
        if e["provider"] == "claude" and e["timestamp"].startswith(month)
    )


def get_daily_summary(date_str: str = None) -> dict:
    """Spend + call counts per provider for a single UTC day."""
    today = date_str or datetime.now(timezone.utc).date().isoformat()
    summary = {p: {"cost": 0.0, "calls": 0}
               for p in ("claude", "gemini", "groq", "openrouter")}
    for e in _read_log():
        if e["timestamp"].startswith(today):
            p = e["provider"]
            summary.setdefault(p, {"cost": 0.0, "calls": 0})
            summary[p]["cost"] += e["cost_usd"]
            summary[p]["calls"] += 1
    summary["total_cost"] = round(sum(v["cost"] for v in summary.values() if isinstance(v, dict)), 4)
    return summary


def get_monthly_summary(month_str: str = None) -> dict:
    """Spend per provider for the current (or given) UTC month."""
    month = month_str or datetime.now(timezone.utc).strftime("%Y-%m")
    summary = {"claude": 0.0, "gemini": 0.0, "groq": 0.0, "openrouter": 0.0}
    for e in _read_log():
        if e["timestamp"].startswith(month):
            summary[e["provider"]] = summary.get(e["provider"], 0.0) + e["cost_usd"]
    summary["total"] = round(sum(summary.values()), 4)
    return summary
