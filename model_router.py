"""
model_router.py — routes each task to the correct AI model based on stakes.

PHILOSOPHY
----------
- Claude Sonnet (paid)  → high-stakes judgment ONLY (scam, eligibility, scoring)
- Free providers        → everything else, tried in a fixed order

Plus:
- Hard daily/monthly budget cap on Claude (downgrade to the free chain).
- Graceful fallback chain when free models fail.
- Every call is logged with which model ran and what it cost.

Free-provider chain (in order, configurable via MODEL_PROVIDER_ORDER):
    1. Groq       — primary
    2. OpenRouter — independent backup (skipped when no API key is set)
    3. Mistral    — free tier (skipped when no API key is set)
    4. Gemini     — last resort; its model ids churn and have broken us twice

A provider that returns 429 (RATE_LIMITED) or a configuration error such as
404 / model-not-found (FAILED) is skipped for the REST OF THE SCAN, so one
bad model name costs one request, not one per operation per candidate.

    Claude fails           → re-raise (fail loud; high stakes)
    ALL free models fail   → Claude Haiku 4.5, but ONLY if budget allows
"""

import json
import os
import re
from threading import RLock
from typing import Optional

import requests

from cost_tracker import log_cost, get_daily_claude_spend, get_monthly_claude_spend
from rate_limiter import (wait_for_quota, seconds_until_available,
                          DailyQuotaExceeded)

# ── Lazy client initialization ────────────────────────────────────────────
# Clients are created on first use so the module imports even when a given
# provider's SDK or key is missing (e.g. running --cost with no Groq key).
_clients = {"anthropic": None, "gemini": None, "groq": None}


def _anthropic():
    if _clients["anthropic"] is None:
        from anthropic import Anthropic
        _clients["anthropic"] = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _clients["anthropic"]


def _gemini(model_name: Optional[str] = None):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel(model_name or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash")


def _groq():
    if _clients["groq"] is None:
        from groq import Groq
        _clients["groq"] = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _clients["groq"]


# ── Task → model mapping (the routing table from the spec) ─────────────────
TASK_ROUTING = {
    # HIGH STAKES — Claude Sonnet ($)
    "scam_detection": "claude",
    "deep_eligibility": "claude",
    "final_scoring": "claude",
    # MEDIUM — Gemini (free)
    "first_pass_filter": "gemini",
    "extract_document_requirements": "gemini",
    "estimate_complexity": "gemini",
    "generate_cover_letter": "gemini",
    "generate_email_subject": "gemini",
    "summarize_opportunity": "gemini",
    "classify_opportunity": "gemini",   # hidden-opportunity detection (free)
    # MECHANICAL — Groq (free). clean_html is deliberately absent: it is
    # done locally in tools.py and makes no API request at all.
    "extract_text": "groq",
    "translate": "groq",
}

# Ordered provider chain, configurable end-to-end via MODEL_PROVIDER_ORDER.
# Default: Groq first (most reliable here), OpenRouter, then Mistral, then
# Gemini last because its model ids churn. Claude is NOT in the default order —
# this project runs at $0, so Claude only participates if explicitly listed
# AND credentialed AND given budget.
DEFAULT_PROVIDER_ORDER = ["groq", "openrouter", "mistral", "gemini"]


def provider_order() -> list:
    raw = os.getenv("MODEL_PROVIDER_ORDER")
    if not raw:
        return list(DEFAULT_PROVIDER_ORDER)
    return [p.strip() for p in raw.split(",") if p.strip()]


# Kept as a module attribute for backward compatibility with existing callers
# and tests; the live order is whatever provider_order() returns.
FREE_CHAIN = list(DEFAULT_PROVIDER_ORDER)

# Human-readable "why this model" used for the transparency log.
ROUTING_REASON = {
    "claude": "paid / only if explicitly configured",
    "groq": "free tier",
    "openrouter": "free tier",
    "mistral": "free tier",
    "gemini": "free tier",
}

# ── Provider states ────────────────────────────────────────────────────────
ENABLED = "ENABLED"            # configured and usable
DISABLED = "DISABLED"          # no credentials — NOT an error
RATE_LIMITED = "RATE_LIMITED"  # 429 this run; skipped for the remainder
FAILED = "FAILED"              # errored this run
AVAILABLE = "AVAILABLE"        # configured, healthy, has run successfully

# Per-scan provider stats + circuit breaker. Reset at the start of each scan.
#
# This is deliberately module-level: ONE latch shared by every operation in a
# scan (clean_html, first_pass_filter, scam_detection, deep_eligibility,
# scoring, ...), not one per call site. Every model call in the project routes
# through call_model() -> _call_free_chain(), so there is a single place where
# the latch is set and a single place where it is honoured.
#
# _STATS_LOCK guards it the way rate_limiter/cost_tracker/database already
# guard their shared state, so the read-modify-write of the counters and the
# latch stays consistent if a scan ever runs operations concurrently. It is an
# RLock because provider_state() -> _stats() nests inside callers that already
# hold it. The lock is never held across a network call.
_provider_stats = {}
_STATS_LOCK = RLock()


# A provider that is over its per-minute quota is not broken — it is busy.
# Blocking on it while an idle provider sits at 0 requests is the worst of
# both worlds, so a wait longer than this is treated as a SOFT failure: skip
# it for THIS call only and move down the chain. Unlike the 404/429 hard
# latches, a soft skip records no failure and never disables the provider.
MAX_PROVIDER_WAIT_SECONDS = float(os.getenv("PROVIDER_MAX_WAIT_SECONDS", "10"))


def _blank_stats() -> dict:
    # ``config_error`` latches a PERMANENT provider fault (404 / model not
    # found / unavailable / invalid model). Unlike an ordinary transient
    # failure, a misconfigured model name cannot fix itself mid-scan, so the
    # provider is skipped for the remainder of the run instead of being
    # retried once per candidate. Kept separate from ``state`` so transient
    # failures stay retryable.
    return {"requests": 0, "successful": 0, "429": 0, "404": 0,
            "failed": 0, "soft_skips": 0, "state": None, "last_error": "",
            "config_error": False}


def reset_provider_stats() -> None:
    """Per-scan reset so the breaker and counters do not leak across runs."""
    with _STATS_LOCK:
        _provider_stats.clear()


def provider_stats(name: str = None):
    with _STATS_LOCK:
        if name is None:
            return {k: dict(v) for k, v in _provider_stats.items()}
        return dict(_provider_stats.setdefault(name, _blank_stats()))


def _stats(name: str) -> dict:
    """The live stats dict for a provider. Mutate it only under _STATS_LOCK."""
    with _STATS_LOCK:
        return _provider_stats.setdefault(name, _blank_stats())


def _record_success(provider: str) -> None:
    """One atomic update so a concurrent reader never sees a half-written row."""
    with _STATS_LOCK:
        s = _stats(provider)
        s["successful"] += 1
        s["state"] = AVAILABLE


def _record_failure(provider: str, exc, task_type: str) -> None:
    """Classify a provider failure and update the breakers, atomically.

    Two per-scan breakers, deliberately distinct:
      * 429            -> RATE_LIMITED  (quota; retrying just burns quota)
      * 404/bad model  -> FAILED + config_error latch (retrying cannot succeed)
    Anything else is transient (5xx, timeout, connection reset) and latches
    nothing, so the provider is retried on the next operation.

    Errors are classified and logged here, never swallowed: the caller still
    falls through to the next provider, and ProvidersUnavailable is still
    raised if every provider is exhausted.
    """
    short = _short_err(exc)
    with _STATS_LOCK:
        s = _stats(provider)
        s["last_error"] = short
        if _is_rate_limit(exc):
            s["429"] += 1
            s["state"] = RATE_LIMITED
            print(f"⚠️  {provider} {task_type}: rate limited — "
                  f"disabling {provider} for the rest of this scan")
            return
        s["failed"] += 1
        s["state"] = FAILED
        if _is_model_not_found(exc):
            s["404"] += 1
            # Permanent: the configured model is wrong, retired, or not
            # available to this account. It will fail identically for every
            # remaining operation in this scan, so latch it instead of
            # burning one request per operation per candidate.
            s["config_error"] = True
            print(f"⛔ {provider} {task_type}: configured model not "
                  f"found/unavailable — check the model name. {short}")
            print(f"   ⛔ disabling {provider} for the rest of this scan "
                  f"(configuration error, not transient)")
        else:
            # Transient (network blip, 5xx, timeout): stays retryable.
            print(f"⚠️  {provider} {task_type}: {short}")


def provider_configured(name: str) -> bool:
    """Credentials present? Absence is DISABLED, never an error."""
    return bool({
        "groq": os.getenv("GROQ_API_KEY"),
        "openrouter": os.getenv("OPENROUTER_API_KEY"),
        "mistral": os.getenv("MISTRAL_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "claude": os.getenv("ANTHROPIC_API_KEY"),
    }.get(name))


def provider_state(name: str) -> str:
    if not provider_configured(name):
        return DISABLED
    s = _stats(name)
    if s["state"] == RATE_LIMITED:
        return RATE_LIMITED
    # A latched configuration fault outranks earlier successes: the provider
    # worked, then its model became unusable, and it stays FAILED this scan.
    if s.get("config_error"):
        return FAILED
    if s["successful"]:
        return AVAILABLE
    if s["state"] == FAILED:
        return FAILED
    return ENABLED


def _is_rate_limit(exc) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code",
                     getattr(exc, "status_code", None))
    return status == 429 or "429" in str(exc) or "rate limit" in str(exc).lower()


# Markers of a PERMANENT provider fault — a wrong, retired or unavailable
# model name. These cannot resolve themselves mid-scan, so they latch.
# Deliberately model-specific: a bare "unavailable" would also match
# "503 service unavailable", which IS transient and must stay retryable.
_MODEL_CONFIG_ERRORS = (
    "model not found", "model_not_found", "no longer available",
    "does not exist", "invalid model", "invalid_model", "unknown model",
    "unsupported model", "model is not available", "model unavailable",
    "is not a valid model", "decommissioned",
)


def _is_model_not_found(exc) -> bool:
    """True for a permanent model/configuration fault (404, bad model name)."""
    status = getattr(getattr(exc, "response", None), "status_code",
                     getattr(exc, "status_code", None))
    text = str(exc).lower()
    if status == 404:
        return True
    if any(m in text for m in _MODEL_CONFIG_ERRORS):
        return True
    # Generic 404/"not found" only when the message is clearly about the
    # model, so a DNS or endpoint blip is not mistaken for a config error.
    return "model" in text and ("not found" in text or "404" in text)


def call_model(task_type: str, prompt: str, system: str = None,
               tools: list = None, max_tokens: int = 2048,
               temperature: float = 0.3) -> dict:
    """Universal entry point. Routes to the correct model for ``task_type``.

    Returns a dict::

        {
          "content": str,
          "model_used": str,
          "task_type": str,
          "tokens_used": {"input": int, "output": int},
          "cost_usd": float,
          "fell_back": bool,
        }
    """
    primary = TASK_ROUTING.get(task_type, "gemini")  # default to free

    # Claude is opt-in only. This project runs at $0, so a Claude-routed task
    # is served by the free chain unless Claude is BOTH credentialed and
    # within budget. Absence of a Claude key is DISABLED, not an error.
    if primary == "claude" and _provider_available("claude"):
        print(f"🤖 [{task_type}] → claude ({ROUTING_REASON['claude']})")
        return _call_claude(prompt, system, tools, max_tokens, temperature,
                            task_type)

    # Everything else — including downgraded Claude tasks — walks the
    # configured provider order.
    return _call_free_chain(prompt, system, max_tokens, temperature, task_type)


def model_health_lines() -> list:
    """Per-provider usage for the scan summary. Never prints credentials."""
    lines = ["🧠 MODEL HEALTH"]
    for name in provider_order() + ["claude"]:
        if name == "claude" and "claude" in provider_order():
            continue
        configured = provider_configured(name)
        s = provider_stats(name)
        state = provider_state(name)
        if not configured:
            lines.append(f"   {name.capitalize():11} configured: NO   "
                         f"status: {DISABLED}")
            continue
        detail = (f"requests {s['requests']}, ok {s['successful']}, "
                  f"429 {s['429']}, 404 {s['404']}, failed {s['failed']}")
        if s.get("soft_skips"):
            # Busy, not broken — these never disabled the provider.
            detail += f", throttle-skipped {s['soft_skips']}"
        lines.append(f"   {name.capitalize():11} configured: YES  "
                     f"status: {state}  •  {detail}")
        if name == "mistral":
            lines.append(f"               model: {mistral_model()}")
        if name == "gemini":
            lines.append(f"               model: "
                         f"{os.getenv('GEMINI_MODEL') or 'gemini-2.5-flash'}")
        if s["last_error"]:
            lines.append(f"               last error: {s['last_error'][:90]}")
    return lines


def log_model_configuration() -> None:
    """Startup log of configured providers + model names. No credentials."""
    order = provider_order()
    print(f"🧠 Provider order: {' → '.join(order)}")
    for name in order:
        if not provider_configured(name):
            print(f"   {name:11} {DISABLED} (no API key configured)")
            continue
        extra = ""
        if name == "mistral":
            extra = f"  model: {mistral_model()}"
        elif name == "gemini":
            extra = f"  model: {os.getenv('GEMINI_MODEL') or 'gemini-2.5-flash'}"
        elif name == "openrouter":
            extra = f"  model: {openrouter_model()}"
        elif name == "groq":
            extra = (f"  model: "
                     f"{os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}")
        print(f"   {name:11} READY{extra}")
    if not provider_configured("claude"):
        print(f"   {'claude':11} {DISABLED} (no API key configured — "
              f"project runs at $0)")


def claude_budget_caps():
    """(daily_cap, monthly_cap) for paid Claude spend, from the environment."""
    return (float(os.getenv("DAILY_CLAUDE_BUDGET_USD", "0.50")),
            float(os.getenv("MONTHLY_CLAUDE_BUDGET_USD", "10.00")))


def has_claude_budget() -> bool:
    """True only if BOTH the daily and monthly Claude caps have headroom.

    A cap of 0 means "no paid calls at all" — ``spend >= 0`` is always true, so
    this correctly returns False rather than reading 0 as "unlimited".
    """
    try:
        daily_cap, monthly_cap = claude_budget_caps()
        return (get_daily_claude_spend() < daily_cap
                and get_monthly_claude_spend() < monthly_cap)
    except Exception as e:
        print(f"⚠️  Could not read Claude budget ({e}) — assuming exhausted.")
        return False


def _short_err(e) -> str:
    """One-line provider error, with the HTTP status when there is one."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    msg = " ".join(str(e).split())
    return f"{status} {msg}"[:140] if status else msg[:140]


def _provider_available(provider: str) -> bool:
    """Configured, and not already rate-limited/misconfigured this scan.

    Two per-scan breakers, deliberately distinct:
      * 429            → RATE_LIMITED  (quota; retrying costs quota)
      * 404/bad model  → FAILED        (config; retrying can never succeed)
    A transient failure latches neither and is retried on the next candidate.
    """
    if provider == "claude":
        # Paid: only if explicitly credentialed AND within budget.
        return provider_configured("claude") and has_claude_budget()
    if not provider_configured(provider):
        return False
    s = _stats(provider)
    return s["state"] != RATE_LIMITED and not s.get("config_error")


def _dispatch_free(provider, prompt, system, max_tokens, temperature, task_type):
    # Resolved by name at call time so the individual _call_* functions stay
    # independently patchable in tests.
    if provider == "groq":
        return _call_groq(prompt, system, max_tokens, temperature, task_type)
    if provider == "openrouter":
        return _call_openrouter(prompt, system, max_tokens, temperature, task_type)
    if provider == "mistral":
        return _call_mistral(prompt, system, max_tokens, temperature, task_type)
    if provider == "gemini":
        return _call_gemini(prompt, system, max_tokens, temperature, task_type)
    if provider == "claude":
        return _call_claude(prompt, system, None, max_tokens, temperature,
                            task_type, model_override=os.getenv(
                                "ANTHROPIC_FALLBACK_MODEL", "claude-haiku-4-5"))
    raise RuntimeError(f"Unknown provider: {provider}")


class ProvidersUnavailable(RuntimeError):
    """Every configured provider was unavailable for this task.

    A TECHNICAL failure — explicitly NOT a judgment about the opportunity.
    Callers must translate this into ANALYSIS_UNAVAILABLE and preserve the
    candidate, never into INELIGIBLE.
    """


def _call_free_chain(prompt, system, max_tokens, temperature, task_type):
    """Walk the configured provider order; one provider's failure never aborts
    the task, and a provider that returns 429 is dropped for the rest of the
    scan instead of being retried on every subsequent call.
    """
    order = provider_order()
    chain = [p for p in order if _provider_available(p)]
    for skipped in (p for p in order if p not in chain):
        state = provider_state(skipped)
        print(f"  ↷ [{task_type}] skipping {skipped} ({state})")

    attempted = []
    throttled = []          # (wait_seconds, provider) soft-skipped THIS call
    for i, provider in enumerate(chain):
        # Re-check immediately before dispatch, not just when the chain was
        # built: under concurrency another operation may have latched this
        # provider in between, and the whole point is that the first failure
        # is the last request anyone sends it.
        if not _provider_available(provider):
            print(f"  ↷ [{task_type}] skipping {provider} "
                  f"({provider_state(provider)})")
            continue

        # Busy is not broken. If this provider would make us sleep off its
        # per-minute quota, prefer a provider that is idle right now. Soft:
        # nothing is recorded as a failure and nothing is latched, so the
        # provider is a first-class candidate again on the very next call.
        wait = _throttle_wait(provider)
        if wait > MAX_PROVIDER_WAIT_SECONDS:
            with _STATS_LOCK:
                _stats(provider)["soft_skips"] += 1
            throttled.append((wait, provider))
            nxt = chain[i + 1] if i + 1 < len(chain) else "nothing left"
            print(f"  ⏭️  [{task_type}] {provider} needs "
                  f"{_wait_str(wait)} of rate-limit sleep — advancing to "
                  f"{nxt} (soft skip, not disabled)")
            continue

        res = _attempt(provider, i, prompt, system, max_tokens, temperature,
                       task_type)
        if res is not None:
            return res
        nxt = chain[i + 1] if i + 1 < len(chain) else "nothing left"
        print(f"   → falling back to {nxt}")
        attempted.append(provider)

    # Everything usable was throttled and nothing was actually tried. Waiting
    # beats failing the candidate, so take the shortest wait rather than
    # raising ProvidersUnavailable.
    if not attempted and throttled:
        wait, provider = min(throttled)
        if wait != float("inf"):
            print(f"⏳ [{task_type}] every provider is rate-limited — waiting "
                  f"{_wait_str(wait)} for {provider}, the soonest available")
            res = _attempt(provider, chain.index(provider), prompt, system,
                           max_tokens, temperature, task_type)
            if res is not None:
                return res
            attempted.append(provider)

    raise ProvidersUnavailable(
        f"all configured providers unavailable for '{task_type}' "
        f"(tried: {', '.join(attempted) or 'none'})")


def _throttle_wait(provider: str) -> float:
    """Seconds this provider would make us sleep. Never raises, never sleeps."""
    try:
        return seconds_until_available(provider)
    except Exception:
        # A probe failure must never stop us from trying the provider.
        return 0.0


def _wait_str(wait: float) -> str:
    return "its full daily quota" if wait == float("inf") else f"{wait:.0f}s"


def _attempt(provider, index, prompt, system, max_tokens, temperature,
             task_type):
    """One dispatch. Returns the result, or None if the provider failed.

    Failures are classified and recorded by _record_failure (429 -> hard
    RATE_LIMITED latch, 404 -> hard config_error latch, anything else
    transient), exactly as before.
    """
    with _STATS_LOCK:
        _stats(provider)["requests"] += 1
    print(f"🤖 [{task_type}] → {provider} "
          f"({ROUTING_REASON.get(provider, 'free')})")
    try:
        res = _dispatch_free(provider, prompt, system, max_tokens,
                             temperature, task_type)
    except Exception as e:
        _record_failure(provider, e, task_type)
        return None
    _record_success(provider)
    res["fell_back"] = index > 0
    return res


# ── Provider implementations ──────────────────────────────────────────────
def _call_claude(prompt, system, tools, max_tokens, temperature, task_type,
                 model_override: str = None) -> dict:
    wait_for_quota("claude")
    model = model_override or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools

    response = _anthropic().messages.create(**kwargs)
    text = "".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    )
    cost = _calculate_claude_cost(response.usage, model)
    log_cost("claude", task_type, response.usage, cost)
    return {
        "content": text,
        "model_used": model,
        "task_type": task_type,
        "tokens_used": {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        },
        "cost_usd": cost,
        "fell_back": bool(model_override),
        "raw_response": response,
    }


def _call_gemini(prompt, system, max_tokens, temperature, task_type) -> dict:
    wait_for_quota("gemini")
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    model = _gemini()
    response = model.generate_content(
        full_prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    # Token counts when exposed by the SDK.
    tin = tout = 0
    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        tin = getattr(meta, "prompt_token_count", 0) or 0
        tout = getattr(meta, "candidates_token_count", 0) or 0
    log_cost("gemini", task_type, {"input": tin, "output": tout}, 0.0)
    return {
        "content": _gemini_text(response),
        "model_used": os.getenv("GEMINI_MODEL") or "gemini-2.5-flash",
        "task_type": task_type,
        "tokens_used": {"input": tin, "output": tout},
        "cost_usd": 0.0,
        "fell_back": False,
    }


def _gemini_text(response) -> str:
    """Safely pull text out of a Gemini response (handles blocked/empty)."""
    try:
        return response.text
    except Exception:
        try:
            parts = response.candidates[0].content.parts
            return "".join(getattr(p, "text", "") for p in parts)
        except Exception:
            return ""


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def openrouter_model() -> str:
    """Default to OpenRouter's auto-router, which outlives individual free
    model ids as they rotate. Never a hardcoded dated model."""
    return os.getenv("OPENROUTER_MODEL") or "openrouter/free"


def _call_openrouter(prompt, system, max_tokens, temperature, task_type) -> dict:
    """OpenAI-compatible chat completion via OpenRouter. Free tier → $0."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    wait_for_quota("openrouter")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    model = openrouter_model()
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional attribution headers; harmless if the API ignores them.
            "HTTP-Referer": "https://github.com/guleaada/opportunitybot",
            "X-Title": "OpportunityBot",
        },
        json={"model": model, "messages": messages,
              "max_tokens": max_tokens, "temperature": temperature},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    # OpenRouter can return HTTP 200 with an error envelope.
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        raise RuntimeError(
            f"{err.get('code', 'error')}: {err.get('message', err)}")

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("no choices returned")
    content = (choices[0].get("message") or {}).get("content") or ""

    usage = data.get("usage") or {}
    tin = usage.get("prompt_tokens", 0) or 0
    tout = usage.get("completion_tokens", 0) or 0
    log_cost("openrouter", task_type, {"input": tin, "output": tout}, 0.0)
    return {
        "content": content,
        "model_used": data.get("model") or model,
        "task_type": task_type,
        "tokens_used": {"input": tin, "output": tout},
        "cost_usd": 0.0,
        "fell_back": False,
    }


MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"


def mistral_model() -> str:
    """Configured Mistral model.

    MISTRAL_MODEL wins if set. The default is a small model documented as
    available on Mistral's free tier — but free-tier eligibility is
    account/model specific, so if it is not available to this account the call
    returns a model-not-found error and the chain moves on. We never silently
    substitute a paid model.
    """
    return os.getenv("MISTRAL_MODEL") or "mistral-small-latest"


def _call_mistral(prompt, system, max_tokens, temperature, task_type) -> dict:
    """Mistral chat completion (OpenAI-compatible shape). Free tier → $0."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not set")
    wait_for_quota("mistral")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    model = mistral_model()
    resp = requests.post(
        MISTRAL_URL,
        headers={"Authorization": f"Bearer {api_key}",     # never logged
                 "Content-Type": "application/json",
                 "Accept": "application/json"},
        json={"model": model, "messages": messages,
              "max_tokens": max_tokens, "temperature": temperature},
        timeout=int(os.getenv("MISTRAL_TIMEOUT", "60")),
    )
    if resp.status_code == 429:
        raise RuntimeError(f"429 rate limited (model {model})")
    if resp.status_code in (400, 404) and "model" in (resp.text or "").lower():
        raise RuntimeError(
            f"model '{model}' not available to this account "
            f"(HTTP {resp.status_code}) — set MISTRAL_MODEL to a model your "
            f"free tier includes")
    resp.raise_for_status()

    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        err = data["error"]
        raise RuntimeError(str(err.get("message", err))[:200])
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("no choices returned")
    content = (choices[0].get("message") or {}).get("content") or ""

    usage = data.get("usage") or {}
    tin = usage.get("prompt_tokens", 0) or 0
    tout = usage.get("completion_tokens", 0) or 0
    log_cost("mistral", task_type, {"input": tin, "output": tout}, 0.0)
    return {
        "content": content,
        "model_used": data.get("model") or model,
        "task_type": task_type,
        "tokens_used": {"input": tin, "output": tout},
        "cost_usd": 0.0,
        "fell_back": False,
    }


def _call_groq(prompt, system, max_tokens, temperature, task_type) -> dict:
    wait_for_quota("groq")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _groq().chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    usage = response.usage
    log_cost("groq", task_type, usage, 0.0)
    return {
        "content": response.choices[0].message.content,
        "model_used": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "task_type": task_type,
        "tokens_used": {
            "input": getattr(usage, "prompt_tokens", 0),
            "output": getattr(usage, "completion_tokens", 0),
        },
        "cost_usd": 0.0,
        "fell_back": False,
    }


# ── Pricing ────────────────────────────────────────────────────────────────
# Per-million-token rates (USD). Verify against current Anthropic pricing.
_CLAUDE_PRICES = {
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-3-5-haiku": (0.80, 4.0),
}


def _calculate_claude_cost(usage, model: str) -> float:
    # Match on prefix so dated model ids resolve to the right tier.
    rate_in, rate_out = 3.0, 15.0
    for key, (ri, ro) in _CLAUDE_PRICES.items():
        if model.startswith(key):
            rate_in, rate_out = ri, ro
            break
    return (usage.input_tokens * rate_in / 1_000_000
            + usage.output_tokens * rate_out / 1_000_000)


# ── JSON helper ─────────────────────────────────────────────────────────────
def extract_json(text: str):
    """Best-effort: pull the first JSON object/array out of a model reply.

    Models sometimes wrap JSON in ```json fences or add prose. Returns the
    parsed object, or ``None`` if nothing parseable is found.
    """
    if not text:
        return None
    # Strip code fences.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fall back to the first properly *balanced* {...} or [...] span, scanning
    # depth (and trying every opener occurrence, not just the first) so stray
    # braces elsewhere in the model's prose can't produce an unbalanced slice
    # or shadow the real JSON that follows.
    for opener, closer in (("{", "}"), ("[", "]")):
        for span in _iter_balanced_spans(candidate, opener, closer):
            try:
                return json.loads(span)
            except json.JSONDecodeError:
                continue
    return None


def _iter_balanced_spans(text: str, opener: str, closer: str):
    """Yield every substring of ``text`` that starts with ``opener`` and is
    depth-balanced against ``closer``, ignoring braces inside quoted strings.
    Tries every occurrence of ``opener`` in order, not just the first."""
    start = text.find(opener)
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    yield text[start:i + 1]
                    break
        start = text.find(opener, start + 1)


# ── Connectivity self-test (used by `main.py --test`) ──────────────────────
def test_providers() -> dict:
    """Ping each provider with a tiny prompt. Returns per-provider status."""
    results = {}
    probes = [
        ("groq", lambda: _call_groq("Reply with exactly: OK", None, 16, 0.0, "extract_text")),
        ("gemini", lambda: _call_gemini("Reply with exactly: OK", None, 16, 0.0, "first_pass_filter")),
        ("claude", lambda: _call_claude("Reply with exactly: OK", None, None, 16, 0.0, "final_scoring")),
    ]
    for name, fn in probes:
        try:
            res = fn()
            results[name] = {
                "ok": True,
                "model": res["model_used"],
                "reply": (res["content"] or "").strip()[:40],
            }
        except Exception as e:
            results[name] = {"ok": False, "error": str(e)}
    return results
