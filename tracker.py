"""
tracker.py — application tracker / mini-CRM (no model calls).

Thin presentation layer over database.py's tracker store, plus a Rich table
renderer used by `main.py --tracker`.
"""

from datetime import datetime, timezone

import database as db

try:
    from rich.console import Console
    from rich.table import Table
    _console = Console()
    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _console = None
    _HAS_RICH = False


def add_application(opportunity: dict, status: str = "interested") -> str:
    return db.add_to_tracker({**opportunity, "status": status})


def set_status(oid: str, status: str, note: str = "") -> bool:
    if status not in db.VALID_STATUSES:
        print(f"⚠️  Unknown status {status!r}. Valid: {', '.join(db.VALID_STATUSES)}")
        return False
    return db.update_tracker(oid, status, note=note)


def _days_left(deadline) -> str:
    if not deadline:
        return "—"
    try:
        d = datetime.fromisoformat(str(deadline)).date()
        return str((d - datetime.now(timezone.utc).date()).days)
    except ValueError:
        return "—"


def show_tracker() -> None:
    tracked = db.all_tracked()
    if not tracked:
        print("📋 Application tracker is empty.")
        return

    rows = sorted(tracked.values(), key=lambda r: r.get("updated", ""), reverse=True)
    if _HAS_RICH:
        table = Table(title="📋 Application Tracker", show_lines=False)
        table.add_column("ID", style="dim", no_wrap=True)
        table.add_column("Opportunity", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Deadline", style="yellow")
        table.add_column("Days", justify="right")
        for r in rows:
            name = r.get("title") or r.get("name") or "—"
            deadline = r.get("deadline") or (r.get("deadline_info") or {}).get("deadline")
            table.add_row(r.get("id", "")[:8], name[:48],
                          r.get("status", "—"), str(deadline or "—"),
                          _days_left(deadline))
        _console.print(table)
    else:
        print("📋 Application Tracker")
        for r in rows:
            name = r.get("title") or r.get("name") or "—"
            deadline = r.get("deadline") or (r.get("deadline_info") or {}).get("deadline")
            print(f"  [{r.get('id','')[:8]}] {name[:48]:48} "
                  f"{r.get('status','—'):14} {str(deadline or '—'):12} "
                  f"days left: {_days_left(deadline)}")
