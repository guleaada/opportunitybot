"""
calendar_sync.py — Google Calendar integration for deadline reminders.

Creates an all-day event on the deadline with reminders at 30/14/7/3/1 days.
Uses OAuth credentials at GOOGLE_CALENDAR_CREDENTIALS_PATH; the first run opens
a browser consent flow and caches a token next to the credentials file.

Degrades gracefully: if google-api-python-client or credentials are missing,
it logs a warning and returns {"added": False, ...} without crashing the scan.
"""

import os
import pickle
from datetime import datetime
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
REMINDER_DAYS = [30, 14, 7, 3, 1]


def _get_service():
    creds_path = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_PATH", "./credentials.json")
    if not Path(creds_path).exists():
        return None, "missing_credentials_file"
    try:
        from google.oauth2.credentials import Credentials  # noqa: F401
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError as e:
        return None, f"missing_dependency: {e}"

    token_path = Path(creds_path).with_name("calendar_token.pickle")
    creds = None
    if token_path.exists():
        with open(token_path, "rb") as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        try:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(token_path, "wb") as f:
                pickle.dump(creds, f)
        except Exception as e:
            return None, f"auth_failed: {e}"

    try:
        return build("calendar", "v3", credentials=creds), None
    except Exception as e:
        return None, f"build_failed: {e}"


def add_to_calendar(data: dict) -> dict:
    """Add a deadline event for an opportunity.

    Expects ``data`` to carry a deadline date at data["deadline"] or
    data["score"]["deadline"] / data["deadline_info"]["deadline"] (ISO date).
    """
    deadline = (
        data.get("deadline")
        or (data.get("deadline_info") or {}).get("deadline")
        or (data.get("score") or {}).get("deadline")
    )
    if not deadline:
        return {"added": False, "reason": "no_deadline"}

    service, err = _get_service()
    if service is None:
        print(f"⚠️  Calendar not configured ({err}) — skipping calendar add.")
        return {"added": False, "reason": err}

    title = data.get("title") or data.get("name") or "Opportunity deadline"
    url = data.get("url", "")
    try:
        # Validate / normalize the date.
        d = datetime.fromisoformat(str(deadline)).date()
    except ValueError:
        return {"added": False, "reason": "bad_deadline_format"}

    event = {
        "summary": f"⏰ Deadline: {title}",
        "description": f"Application deadline for {title}\n{url}",
        "start": {"date": d.isoformat()},
        "end": {"date": d.isoformat()},
        "reminders": {
            "useDefault": False,
            "overrides": [
                {"method": "popup", "minutes": days * 24 * 60}
                for days in REMINDER_DAYS
            ],
        },
    }
    try:
        created = service.events().insert(calendarId="primary", body=event).execute()
        print(f"📅 Calendar event created: {created.get('htmlLink')}")
        return {"added": True, "link": created.get("htmlLink")}
    except Exception as e:
        print(f"⚠️  Calendar insert failed: {e}")
        return {"added": False, "reason": str(e)}
