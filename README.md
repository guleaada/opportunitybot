# OpportunityBot 🎯

A personal, multi-model AI agent that automatically finds international
**fellowships, scholarships, and programs** matching your profile, filters out
**scams and closed opportunities**, and notifies you only about genuine, open
opportunities scoring **7/10 or higher**.

Built for Gulilat Kasiye Worku — Ethiopian AI developer, working professional,
based in Vientiane, Lao PDR — but the profile is fully editable in
`profile.py`.

---

## 1. What it is

OpportunityBot runs a daily scan across trusted scholarship sources (DAAD,
Chevening, Fulbright, MEXT, Erasmus, Commonwealth, and reputable aggregators),
deeply analyzes each candidate for **legitimacy**, **eligibility**, and **fit**,
and emails/Telegrams you a ranked report. Deadlines for top matches are pushed
to Google Calendar with 30/14/7/3/1-day reminders.

## 2. The triage architecture (why it's cheap)

The whole point is **cost efficiency**: use your paid Claude credit only where
judgment really matters. Each task is routed by `model_router.py`:

| Task | Model | Why |
|---|---|---|
| Scam / legitimacy detection | **Claude Sonnet** ($) | Highest stakes — a wrong call wastes your application time |
| Deep eligibility (final pass) | **Claude Sonnet** ($) | Nuanced judgment |
| Final scoring + reasoning | **Claude Sonnet** ($) | Final decision quality |
| First-pass filter | **Gemini Flash** (free) | High volume, simple |
| Document extraction | **Gemini Flash** (free) | Structured extraction |
| Complexity estimate | **Gemini Flash** (free) | Estimation |
| Cover-letter draft | **Gemini Flash** (free) | Creative, revisable |
| Email subject line | **Gemini Flash** (free) | Trivial |
| HTML→text cleanup | **Groq Llama 3.3** (free) | Mechanical |
| Translation | **Groq Llama 3.3** (free) | Mechanical |

The pipeline runs **cheap models first** so Groq/Gemini discard ~80% of
candidates before any Claude token is spent. Typical day: ~50 discovered → ~3
reach Claude → **~$0.15/day**.

**Fallback chain:** Claude failure → fail loud (no silent downgrade). Gemini
failure → Groq. Groq failure → Gemini. All free models fail → Claude **Haiku**
as last resort. A **hard daily/monthly budget cap** downgrades Claude → Gemini
if you'd otherwise overspend.

## 3. Setup

> **Python 3.11+ recommended.** The code is written to also run on 3.9+, but
> the project targets 3.11+.

```bash
# Clone / enter the project
cd opportunitybot

# Create + activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create your .env from the template and fill in keys
cp .env.example .env
$EDITOR .env
```

### How to get each API key

| Key | Where | Cost |
|---|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | Paid — **~$5 lasts 3+ weeks** at default settings |
| `GEMINI_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | **FREE**, no credit card |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | **FREE**, no credit card |
| `EMAIL_APP_PASSWORD` | [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords) (needs 2FA on) | Free |
| `GOOGLE_CSE_API_KEY` + `GOOGLE_CSE_ID` | [console.cloud.google.com](https://console.cloud.google.com) → enable **Custom Search API** → create a Programmable Search Engine at [programmablesearchengine.google.com](https://programmablesearchengine.google.com) (set it to search the entire web) | Free tier: 100 queries/day |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | (optional) [@BotFather](https://t.me/botfather) | Free |
| `GOOGLE_CALENDAR_CREDENTIALS_PATH` | (optional) Google Cloud → OAuth client → download `credentials.json` | Free |

Any missing optional credential is **skipped gracefully** — the bot still runs.

## 4. First run

```bash
# Verify all three model providers connect
python main.py --test

# Analyze a single page end-to-end (shows which model ran each step)
python main.py --url https://www.chevening.org/scholarships/who-can-apply/

# See spend (should be ~$0 on first runs)
python main.py --cost

# Full scan (needs Google CSE keys to discover; otherwise use --url)
python main.py --scan

# Run scheduled daily at DAILY_SCAN_TIME
python main.py --daemon
```

## 5. Cost expectations

At default settings (`MAX_OPPORTUNITIES_PER_DAY=15`,
`DAILY_CLAUDE_BUDGET_USD=0.50`):

- **~$0.10–$0.30 per daily scan** in Claude spend (Gemini + Groq are free).
- **~$3–$9/month** if you scan every day.
- A `$5` Anthropic credit comfortably covers **3+ weeks** of daily scans.

All spend is logged to `data/cost_log.json` and shown in every report and via
`python main.py --cost`.

## 6. Customizing budget caps

Edit these in `.env`:

```
DAILY_CLAUDE_BUDGET_USD=0.50     # bot downgrades Claude→Gemini past this/day
MONTHLY_CLAUDE_BUDGET_USD=10.00  # and past this/month
MIN_SCORE_TO_NOTIFY=7            # only notify on matches >= this score
MAX_OPPORTUNITIES_PER_DAY=15     # cap deep-analysis volume (caps Claude calls)
DAILY_SCAN_TIME=08:00            # daemon schedule
```

## 7. File map

| File | One-liner |
|---|---|
| `main.py` | Runner + CLI + scan pipeline + report builder |
| `model_router.py` | ★ Routes each task to Claude/Gemini/Groq with budget caps + fallback |
| `rate_limiter.py` | ★ Respects free-tier RPM/RPD limits, queues requests |
| `cost_tracker.py` | ★ Logs spend per provider; daily/monthly rollups |
| `tools.py` | The agent's tool surface; delegates to the router by task type |
| `search.py` | Google Custom Search + cached URL fetch (24h) |
| `checker.py` | Deadline, first-pass filter, legitimacy, eligibility |
| `scorer.py` | Final 0-10 scoring with reasoning (Claude) |
| `profile.py` | Your hardcoded profile |
| `known_scams.py` | Scam/fee-trap blocklist + red-flag heuristics |
| `source_whitelist.py` | Trusted sources with quality scores + tuned queries |
| `database.py` | JSON memory: seen / tracker / watchlist / activity log |
| `notifier.py` | Email (Gmail SMTP) + Telegram notifications |
| `calendar_sync.py` | Google Calendar deadline events with reminders |
| `cover_letter.py` | Motivation-letter draft generator (Gemini) |
| `tracker.py` | Application tracker / mini-CRM display |

## 8. Honesty rule

If a model can't determine a deadline, eligibility, or legitimacy, the item is
flagged **`unknown`** and **skipped** rather than guessed. Better a missed
maybe than a wasted application.
