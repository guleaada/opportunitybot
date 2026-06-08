"""
notifier.py — email (SMTP/Gmail) + Telegram notifications.

Both channels degrade gracefully: if credentials are missing, the report is
printed to stdout and the missing channel is skipped with a warning.
"""

import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests


def send_email(subject: str, body: str) -> dict:
    sender = os.getenv("EMAIL_FROM")
    password = os.getenv("EMAIL_APP_PASSWORD")
    recipient = os.getenv("EMAIL_TO", sender)
    if not sender or not password:
        print("⚠️  EMAIL_FROM / EMAIL_APP_PASSWORD not set — skipping email.")
        return {"sent": False, "reason": "missing_credentials"}

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls(context=context)
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        print(f"📧 Email sent to {recipient}")
        return {"sent": True, "channel": "email"}
    except Exception as e:
        print(f"⚠️  Email send failed: {e}")
        return {"sent": False, "reason": str(e)}


def send_telegram(text: str) -> dict:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return {"sent": False, "reason": "missing_credentials"}
    try:
        # Telegram caps messages at 4096 chars.
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text[:4096],
                  "disable_web_page_preview": False},
            timeout=20,
        )
        resp.raise_for_status()
        print("📱 Telegram message sent")
        return {"sent": True, "channel": "telegram"}
    except Exception as e:
        print(f"⚠️  Telegram send failed: {e}")
        return {"sent": False, "reason": str(e)}


def send_notification(report: str, channel: str = "all", subject: str = None) -> dict:
    """Send ``report`` via the requested channel(s).

    channel: "email" | "telegram" | "all". Always prints the report too.
    """
    subject = subject or "🎯 OpportunityBot Daily Report"
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60 + "\n")

    results = {}
    if channel in ("email", "all"):
        results["email"] = send_email(subject, report)
    if channel in ("telegram", "all"):
        results["telegram"] = send_telegram(report)
    return results
