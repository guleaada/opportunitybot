"""
model_router.py — routes each task to the correct AI model based on stakes.

PHILOSOPHY
----------
- Claude Sonnet (paid)  → high-stakes judgment ONLY
- Free providers        → everything else, tried in a fixed order

Free-provider chain:
    1. Groq
    2. OpenRouter
    3. Mistral
    4. Gemini

Provider model names are configuration-driven through environment variables:

    GROQ_MODEL
    OPENROUTER_MODEL
    MISTRAL_MODEL
    GEMINI_MODEL
    ANTHROPIC_MODEL
    ANTHROPIC_FALLBACK_MODEL

IMPORTANT
---------
There are NO hardcoded Groq model IDs in this file.

A provider returning 429 or a model/configuration error is disabled
for the REST OF THE SCAN.

Claude failures remain loud for high-stakes tasks.

If every free provider fails, Claude Haiku is used only when the
Claude budget allows it.
"""

import json
import os
import re
from threading import RLock
from typing import Optional

import requests

from cost_tracker import (
    log_cost,
    get_daily_claude_spend,
    get_monthly_claude_spend,
)

from rate_limiter import (
    wait_for_quota,
    seconds_until_available,
)


# ═══════════════════════════════════════════════════════════════════════════
# LAZY CLIENT INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════

_clients = {
    "anthropic": None,
    "gemini": None,
    "groq": None,
}


def _anthropic():
    if _clients["anthropic"] is None:
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set"
            )

        _clients["anthropic"] = Anthropic(
            api_key=api_key
        )

    return _clients["anthropic"]


def _gemini(model_name: Optional[str] = None):
    """
    Create the Gemini client using the current google-genai SDK.

    The model name is passed to generate_content().
    It is NOT used to construct a model object.
    """

    from google import genai

    if _clients["gemini"] is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set"
            )

        _clients["gemini"] = genai.Client(
            api_key=api_key
        )

    return _clients["gemini"]


def _groq():
    if _clients["groq"] is None:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set"
            )

        _clients["groq"] = Groq(
            api_key=api_key
        )

    return _clients["groq"]


# ═══════════════════════════════════════════════════════════════════════════
# TASK → PROVIDER MAPPING
# ═══════════════════════════════════════════════════════════════════════════

TASK_ROUTING = {
    # HIGH STAKES — Claude
    "scam_detection": "claude",
    "deep_eligibility": "claude",
    "final_scoring": "claude",

    # MEDIUM — Gemini
    "first_pass_filter": "gemini",
    "check_deadline": "gemini",
    "extract_document_requirements": "gemini",
    "estimate_complexity": "gemini",
    "generate_cover_letter": "gemini",
    "generate_email_subject": "gemini",
    "summarize_opportunity": "gemini",
    "classify_opportunity": "gemini",

    # MECHANICAL — Groq
    "extract_text": "groq",
    "translate": "groq",
}


# ═══════════════════════════════════════════════════════════════════════════
# PROVIDER ORDER
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_PROVIDER_ORDER = [
    "groq",
    "openrouter",
    "mistral",
    "gemini",
]


def provider_order() -> list:
    """
    Return configured free-provider order.

    MODEL_PROVIDER_ORDER can override the default.

    Example:
        MODEL_PROVIDER_ORDER=groq,openrouter,mistral,gemini
    """

    raw = os.getenv("MODEL_PROVIDER_ORDER")

    if not raw:
        return list(DEFAULT_PROVIDER_ORDER)

    return [
        p.strip().lower()
        for p in raw.split(",")
        if p.strip()
    ]


# Backward compatibility
FREE_CHAIN = list(DEFAULT_PROVIDER_ORDER)


# ═══════════════════════════════════════════════════════════════════════════
# ROUTING REASONS
# ═══════════════════════════════════════════════════════════════════════════

ROUTING_REASON = {
    "claude": "paid / high-stakes only",
    "groq": "free tier",
    "openrouter": "free tier",
    "mistral": "free tier",
    "gemini": "free tier",
}


# ═══════════════════════════════════════════════════════════════════════════
# PROVIDER STATES
# ═══════════════════════════════════════════════════════════════════════════

ENABLED = "ENABLED"
DISABLED = "DISABLED"
RATE_LIMITED = "RATE_LIMITED"
FAILED = "FAILED"
AVAILABLE = "AVAILABLE"


# ═══════════════════════════════════════════════════════════════════════════
# PER-SCAN PROVIDER STATS / CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════

_provider_stats = {}
_STATS_LOCK = RLock()

MAX_PROVIDER_WAIT_SECONDS = float(
    os.getenv(
        "PROVIDER_MAX_WAIT_SECONDS",
        "10",
    )
)


def _blank_stats() -> dict:
    return {
        "requests": 0,
        "successful": 0,
        "429": 0,
        "404": 0,
        "failed": 0,
        "soft_skips": 0,
        "state": None,
        "last_error": "",
        "config_error": False,
    }


def reset_provider_stats() -> None:
    """Reset provider stats at the beginning of a scan."""

    with _STATS_LOCK:
        _provider_stats.clear()


def provider_stats(name: str = None):
    with _STATS_LOCK:
        if name is None:
            return {
                k: dict(v)
                for k, v in _provider_stats.items()
            }

        return dict(
            _provider_stats.setdefault(
                name,
                _blank_stats(),
            )
        )


def _stats(name: str) -> dict:
    """Return live stats dictionary."""

    return _provider_stats.setdefault(
        name,
        _blank_stats(),
    )


def _record_success(provider: str) -> None:
    with _STATS_LOCK:
        s = _stats(provider)
        s["successful"] += 1
        s["state"] = AVAILABLE


def _record_failure(
    provider: str,
    exc,
    task_type: str,
) -> None:

    short = _short_err(exc)

    with _STATS_LOCK:
        s = _stats(provider)

        s["last_error"] = short

        if _is_rate_limit(exc):
            s["429"] += 1
            s["state"] = RATE_LIMITED

            print(
                f"⚠️  {provider} {task_type}: "
                f"rate limited — disabling {provider} "
                f"for the rest of this scan"
            )

            return

        s["failed"] += 1
        s["state"] = FAILED

        if _is_model_not_found(exc):
            s["404"] += 1
            s["config_error"] = True

            print(
                f"⛔ {provider} {task_type}: "
                f"configured model not found/unavailable."
            )

            print(
                f"   ⛔ disabling {provider} for the "
                f"rest of this scan"
            )

        else:
            print(
                f"⚠️  {provider} {task_type}: {short}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# PROVIDER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

def provider_configured(name: str) -> bool:
    """
    Check whether provider has the required API key.

    Model configuration is separately validated by each provider.
    """

    keys = {
        "groq": "GROQ_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }

    key_name = keys.get(name)

    if not key_name:
        return False

    return bool(
        os.getenv(key_name)
    )


def provider_state(name: str) -> str:

    if not provider_configured(name):
        return DISABLED

    s = provider_stats(name)

    if s["state"] == RATE_LIMITED:
        return RATE_LIMITED

    if s.get("config_error"):
        return FAILED

    if s["successful"]:
        return AVAILABLE

    if s["state"] == FAILED:
        return FAILED

    return ENABLED


# ═══════════════════════════════════════════════════════════════════════════
# ERROR CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def _is_rate_limit(exc) -> bool:

    status = getattr(
        getattr(exc, "response", None),
        "status_code",
        getattr(
            exc,
            "status_code",
            None,
        ),
    )

    text = str(exc).lower()

    return (
        status == 429
        or "429" in text
        or "rate limit" in text
        or "rate_limit" in text
        or "too many requests" in text
    )


_MODEL_CONFIG_ERRORS = (
    "model not found",
    "model_not_found",
    "no longer available",
    "does not exist",
    "invalid model",
    "invalid_model",
    "unknown model",
    "unsupported model",
    "model is not available",
    "model unavailable",
    "is not a valid model",
    "decommissioned",
    "model not supported",
)


def _is_model_not_found(exc) -> bool:
    """Detect permanent model/configuration errors."""

    status = getattr(
        getattr(exc, "response", None),
        "status_code",
        getattr(
            exc,
            "status_code",
            None,
        ),
    )

    text = str(exc).lower()

    if status == 404:
        return True

    if any(
        marker in text
        for marker in _MODEL_CONFIG_ERRORS
    ):
        return True

    if (
        "model" in text
        and (
            "not found" in text
            or "404" in text
        )
    ):
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSAL MODEL ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def call_model(
    task_type: str,
    prompt: str,
    system: str = None,
    tools: list = None,
    max_tokens: int = 2048,
    temperature: float = 0.3,
) -> dict:

    primary = TASK_ROUTING.get(
        task_type,
        "gemini",
    )

    # High-stakes tasks use Claude when configured and budget allows.
    if (
        primary == "claude"
        and _provider_available("claude")
    ):

        print(
            f"🤖 [{task_type}] → claude "
            f"({ROUTING_REASON['claude']})"
        )

        return _call_claude(
            prompt,
            system,
            tools,
            max_tokens,
            temperature,
            task_type,
        )

    # All normal tasks use the free provider chain.
    return _call_free_chain(
        prompt,
        system,
        max_tokens,
        temperature,
        task_type,
    )


# ═══════════════════════════════════════════════════════════════════════════
# MODEL HEALTH
# ═══════════════════════════════════════════════════════════════════════════

def model_health_lines() -> list:

    lines = ["🧠 MODEL HEALTH"]

    names = provider_order()

    if "claude" not in names:
        names.append("claude")

    for name in names:

        configured = provider_configured(name)
        s = provider_stats(name)
        state = provider_state(name)

        if not configured:
            lines.append(
                f"   {name.capitalize():11} "
                f"configured: NO   "
                f"status: {DISABLED}"
            )
            continue

        detail = (
            f"requests {s['requests']}, "
            f"ok {s['successful']}, "
            f"429 {s['429']}, "
            f"404 {s['404']}, "
            f"failed {s['failed']}"
        )

        if s.get("soft_skips"):
            detail += (
                f", throttle-skipped "
                f"{s['soft_skips']}"
            )

        lines.append(
            f"   {name.capitalize():11} "
            f"configured: YES  "
            f"status: {state}  •  {detail}"
        )

        if name == "groq":
            lines.append(
                f"               model: "
                f"{groq_model() or 'NOT SET — set GROQ_MODEL'}"
            )

        elif name == "openrouter":
            lines.append(
                f"               model: "
                f"{openrouter_model()}"
            )

        elif name == "mistral":
            lines.append(
                f"               model: "
                f"{mistral_model()}"
            )

        elif name == "gemini":
            lines.append(
                f"               model: "
                f"{gemini_model()}"
            )

        elif name == "claude":
            lines.append(
                f"               model: "
                f"{anthropic_model()}"
            )

        if s["last_error"]:
            lines.append(
                f"               last error: "
                f"{s['last_error'][:90]}"
            )

    return lines


# ═══════════════════════════════════════════════════════════════════════════
# STARTUP CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

def log_model_configuration() -> None:

    order = provider_order()

    print(
        f"🧠 Provider order: "
        f"{' → '.join(order)}"
    )

    for name in order:

        if not provider_configured(name):

            print(
                f"   {name:11} {DISABLED} "
                f"(no API key configured)"
            )

            continue

        if name == "groq":

            model = groq_model()

            if model:
                print(
                    f"   {name:11} READY  "
                    f"model: {model}"
                )
            else:
                print(
                    f"   {name:11} READY  "
                    f"model: NOT SET — "
                    f"set GROQ_MODEL"
                )

        elif name == "gemini":

            print(
                f"   {name:11} READY  "
                f"model: {gemini_model()}"
            )

        elif name == "openrouter":

            print(
                f"   {name:11} READY  "
                f"model: {openrouter_model()}"
            )

        elif name == "mistral":

            print(
                f"   {name:11} READY  "
                f"model: {mistral_model()}"
            )

        else:

            print(
                f"   {name:11} READY"
            )

    if not provider_configured("claude"):

        print(
            f"   {'claude':11} {DISABLED} "
            f"(no API key configured — "
            f"project runs at $0)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

def groq_model() -> str:
    """
    Groq model from repository/environment configuration.

    IMPORTANT:
    There is deliberately NO hardcoded Groq model fallback.

    Set:

        GROQ_MODEL=openai/gpt-oss-120b

    or another model currently available to the account.
    """

    return (
        os.getenv("GROQ_MODEL")
        or ""
    ).strip()


def gemini_model() -> str:
    """
    Gemini model from configuration.

    Repository variable currently expected:

        GEMINI_MODEL=gemini-2.5-flash
    """

    return (
        os.getenv("GEMINI_MODEL")
        or "gemini-2.5-flash"
    ).strip()


def openrouter_model() -> str:
    """
    OpenRouter model from configuration.

    If OPENROUTER_MODEL is not set, use the free router.
    """

    return (
        os.getenv("OPENROUTER_MODEL")
        or "openrouter/free"
    ).strip()


def mistral_model() -> str:
    """
    Mistral model from configuration.
    """

    return (
        os.getenv("MISTRAL_MODEL")
        or "mistral-small-latest"
    ).strip()


def anthropic_model() -> str:
    """
    Claude primary model from configuration.
    """

    return (
        os.getenv("ANTHROPIC_MODEL")
        or "claude-sonnet-4-5"
    ).strip()


def anthropic_fallback_model() -> str:
    """
    Claude fallback model from configuration.
    """

    return (
        os.getenv("ANTHROPIC_FALLBACK_MODEL")
        or "claude-haiku-4-5"
    ).strip()


# ═══════════════════════════════════════════════════════════════════════════
# CLAUDE BUDGET
# ═══════════════════════════════════════════════════════════════════════════

def claude_budget_caps():

    return (
        float(
            os.getenv(
                "DAILY_CLAUDE_BUDGET_USD",
                "0.50",
            )
        ),
        float(
            os.getenv(
                "MONTHLY_CLAUDE_BUDGET_USD",
                "10.00",
            )
        ),
    )


def has_claude_budget() -> bool:

    try:

        daily_cap, monthly_cap = (
            claude_budget_caps()
        )

        return (
            get_daily_claude_spend()
            < daily_cap
            and
            get_monthly_claude_spend()
            < monthly_cap
        )

    except Exception as e:

        print(
            f"⚠️  Could not read Claude budget "
            f"({e}) — assuming exhausted."
        )

        return False


# ═══════════════════════════════════════════════════════════════════════════
# ERROR HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _short_err(e) -> str:

    status = getattr(
        getattr(e, "response", None),
        "status_code",
        None,
    )

    msg = " ".join(
        str(e).split()
    )

    if status:
        return (
            f"{status} {msg}"
        )[:140]

    return msg[:140]


# ═══════════════════════════════════════════════════════════════════════════
# PROVIDER AVAILABILITY
# ═══════════════════════════════════════════════════════════════════════════

def _provider_available(
    provider: str,
) -> bool:

    if provider == "claude":

        return (
            provider_configured("claude")
            and has_claude_budget()
        )

    if not provider_configured(provider):
        return False

    # Configuration-level model validation.
    if provider == "groq" and not groq_model():
        return False

    if provider == "gemini" and not gemini_model():
        return False

    if provider == "openrouter" and not openrouter_model():
        return False

    if provider == "mistral" and not mistral_model():
        return False

    s = provider_stats(provider)

    return (
        s["state"] != RATE_LIMITED
        and not s.get("config_error")
    )


# ═══════════════════════════════════════════════════════════════════════════
# DISPATCH
# ═══════════════════════════════════════════════════════════════════════════

def _dispatch_free(
    provider,
    prompt,
    system,
    max_tokens,
    temperature,
    task_type,
):

    if provider == "groq":

        return _call_groq(
            prompt,
            system,
            max_tokens,
            temperature,
            task_type,
        )

    if provider == "openrouter":

        return _call_openrouter(
            prompt,
            system,
            max_tokens,
            temperature,
            task_type,
        )

    if provider == "mistral":

        return _call_mistral(
            prompt,
            system,
            max_tokens,
            temperature,
            task_type,
        )

    if provider == "gemini":

        return _call_gemini(
            prompt,
            system,
            max_tokens,
            temperature,
            task_type,
        )

    raise RuntimeError(
        f"Unknown provider: {provider}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# PROVIDERS UNAVAILABLE
# ═══════════════════════════════════════════════════════════════════════════

class ProvidersUnavailable(RuntimeError):
    """
    Every configured free provider was unavailable.

    This is a technical failure.

    Callers must NOT convert this into INELIGIBLE.
    """


# ═══════════════════════════════════════════════════════════════════════════
# FREE PROVIDER CHAIN
# ═══════════════════════════════════════════════════════════════════════════

def _call_free_chain(
    prompt,
    system,
    max_tokens,
    temperature,
    task_type,
):

    order = provider_order()

    chain = [
        provider
        for provider in order
        if _provider_available(provider)
    ]

    for skipped in (
        p
        for p in order
        if p not in chain
    ):

        state = provider_state(skipped)

        # Special message for missing Groq model.
        if (
            skipped == "groq"
            and provider_configured("groq")
            and not groq_model()
        ):

            print(
                f"  ↷ [{task_type}] skipping groq "
                f"(GROQ_MODEL is not set)"
            )

        else:

            print(
                f"  ↷ [{task_type}] "
                f"skipping {skipped} "
                f"({state})"
            )

    attempted = []
    throttled = []

    for i, provider in enumerate(chain):

        if not _provider_available(provider):

            print(
                f"  ↷ [{task_type}] "
                f"skipping {provider} "
                f"({provider_state(provider)})"
            )

            continue

        wait = _throttle_wait(provider)

        if wait > MAX_PROVIDER_WAIT_SECONDS:

            with _STATS_LOCK:
                _stats(provider)["soft_skips"] += 1

            throttled.append(
                (wait, provider)
            )

            nxt = (
                chain[i + 1]
                if i + 1 < len(chain)
                else "nothing left"
            )

            print(
                f"  ⏭️  [{task_type}] {provider} "
                f"needs {_wait_str(wait)} of "
                f"rate-limit sleep — advancing "
                f"to {nxt} "
                f"(soft skip, not disabled)"
            )

            continue

        res = _attempt(
            provider,
            i,
            prompt,
            system,
            max_tokens,
            temperature,
            task_type,
        )

        if res is not None:
            return res

        nxt = (
            chain[i + 1]
            if i + 1 < len(chain)
            else "nothing left"
        )

        print(
            f"   → falling back to {nxt}"
        )

        attempted.append(provider)

    # If all providers were throttled, try the soonest one.
    if not attempted and throttled:

        wait, provider = min(
            throttled,
            key=lambda item: item[0],
        )

        if wait != float("inf"):

            print(
                f"⏳ [{task_type}] every provider "
                f"is rate-limited — waiting "
                f"{_wait_str(wait)} for {provider}, "
                f"the soonest available"
            )

            res = _attempt(
                provider,
                chain.index(provider),
                prompt,
                system,
                max_tokens,
                temperature,
                task_type,
            )

            if res is not None:
                return res

            attempted.append(provider)

    raise ProvidersUnavailable(
        f"all configured providers unavailable "
        f"for '{task_type}' "
        f"(tried: "
        f"{', '.join(attempted) or 'none'})"
    )


# ═══════════════════════════════════════════════════════════════════════════
# THROTTLING
# ═══════════════════════════════════════════════════════════════════════════

def _throttle_wait(
    provider: str,
) -> float:

    try:

        return seconds_until_available(
            provider
        )

    except Exception:

        return 0.0


def _wait_str(
    wait: float,
) -> str:

    if wait == float("inf"):
        return "its full daily quota"

    return f"{wait:.0f}s"


# ═══════════════════════════════════════════════════════════════════════════
# PROVIDER ATTEMPT
# ═══════════════════════════════════════════════════════════════════════════

def _attempt(
    provider,
    index,
    prompt,
    system,
    max_tokens,
    temperature,
    task_type,
):

    with _STATS_LOCK:
        _stats(provider)["requests"] += 1

    print(
        f"🤖 [{task_type}] → {provider} "
        f"({ROUTING_REASON.get(provider, 'free')})"
    )

    try:

        res = _dispatch_free(
            provider,
            prompt,
            system,
            max_tokens,
            temperature,
            task_type,
        )

    except Exception as e:

        _record_failure(
            provider,
            e,
            task_type,
        )

        return None

    _record_success(provider)

    res["fell_back"] = index > 0

    return res


# ═══════════════════════════════════════════════════════════════════════════
# CLAUDE
# ═══════════════════════════════════════════════════════════════════════════

def _call_claude(
    prompt,
    system,
    tools,
    max_tokens,
    temperature,
    task_type,
    model_override: str = None,
) -> dict:

    wait_for_quota("claude")

    model = (
        model_override
        or anthropic_model()
    )

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    if system:
        kwargs["system"] = system

    if tools:
        kwargs["tools"] = tools

    response = _anthropic().messages.create(
        **kwargs
    )

    text = "".join(
        block.text
        for block in response.content
        if getattr(
            block,
            "type",
            None,
        ) == "text"
    )

    cost = _calculate_claude_cost(
        response.usage,
        model,
    )

    log_cost(
        "claude",
        task_type,
        response.usage,
        cost,
    )

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


# ═══════════════════════════════════════════════════════════════════════════
# GEMINI
# ═══════════════════════════════════════════════════════════════════════════

def _call_gemini(
    prompt,
    system,
    max_tokens,
    temperature,
    task_type,
) -> dict:

    wait_for_quota("gemini")

    full_prompt = (
        f"{system}\n\n{prompt}"
        if system
        else prompt
    )

    model_name = gemini_model()

    if not model_name:
        raise RuntimeError(
            "GEMINI_MODEL is not set"
        )

    client = _gemini()

    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    tin = 0
    tout = 0

    meta = getattr(
        response,
        "usage_metadata",
        None,
    )

    if meta is not None:

        tin = (
            getattr(
                meta,
                "prompt_token_count",
                0,
            )
            or 0
        )

        tout = (
            getattr(
                meta,
                "candidates_token_count",
                0,
            )
            or 0
        )

    log_cost(
        "gemini",
        task_type,
        {
            "input": tin,
            "output": tout,
        },
        0.0,
    )

    return {
        "content": _gemini_text(response),
        "model_used": model_name,
        "task_type": task_type,
        "tokens_used": {
            "input": tin,
            "output": tout,
        },
        "cost_usd": 0.0,
        "fell_back": False,
    }


def _gemini_text(
    response,
) -> str:

    try:

        text = response.text

        if text:
            return text

    except Exception:
        pass

    try:

        parts = (
            response
            .candidates[0]
            .content
            .parts
        )

        return "".join(
            getattr(
                part,
                "text",
                "",
            )
            for part in parts
        )

    except Exception:

        return ""


# ═══════════════════════════════════════════════════════════════════════════
# OPENROUTER
# ═══════════════════════════════════════════════════════════════════════════

OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)


def _call_openrouter(
    prompt,
    system,
    max_tokens,
    temperature,
    task_type,
) -> dict:

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set"
        )

    wait_for_quota("openrouter")

    messages = []

    if system:

        messages.append(
            {
                "role": "system",
                "content": system,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    model = openrouter_model()

    if not model:
        raise RuntimeError(
            "OPENROUTER_MODEL is not set"
        )

    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": (
                f"Bearer {api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
            "HTTP-Referer": (
                "https://github.com/"
                "guleaada/opportunitybot"
            ),
            "X-Title": "OpportunityBot",
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=60,
    )

    resp.raise_for_status()

    data = resp.json()

    if (
        isinstance(data, dict)
        and data.get("error")
    ):

        err = data["error"]

        raise RuntimeError(
            f"{err.get('code', 'error')}: "
            f"{err.get('message', err)}"
        )

    choices = data.get("choices") or []

    if not choices:
        raise RuntimeError(
            "no choices returned"
        )

    content = (
        choices[0]
        .get("message", {})
        .get("content")
        or ""
    )

    usage = data.get("usage") or {}

    tin = (
        usage.get(
            "prompt_tokens",
            0,
        )
        or 0
    )

    tout = (
        usage.get(
            "completion_tokens",
            0,
        )
        or 0
    )

    log_cost(
        "openrouter",
        task_type,
        {
            "input": tin,
            "output": tout,
        },
        0.0,
    )

    return {
        "content": content,
        "model_used": (
            data.get("model")
            or model
        ),
        "task_type": task_type,
        "tokens_used": {
            "input": tin,
            "output": tout,
        },
        "cost_usd": 0.0,
        "fell_back": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MISTRAL
# ═══════════════════════════════════════════════════════════════════════════

MISTRAL_URL = (
    "https://api.mistral.ai/v1/chat/completions"
)


def _call_mistral(
    prompt,
    system,
    max_tokens,
    temperature,
    task_type,
) -> dict:

    api_key = os.getenv(
        "MISTRAL_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY not set"
        )

    wait_for_quota("mistral")

    messages = []

    if system:

        messages.append(
            {
                "role": "system",
                "content": system,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    model = mistral_model()

    if not model:
        raise RuntimeError(
            "MISTRAL_MODEL is not set"
        )

    resp = requests.post(
        MISTRAL_URL,
        headers={
            "Authorization": (
                f"Bearer {api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
            "Accept": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=int(
            os.getenv(
                "MISTRAL_TIMEOUT",
                "60",
            )
        ),
    )

    if resp.status_code == 429:

        raise RuntimeError(
            f"429 rate limited "
            f"(model {model})"
        )

    if (
        resp.status_code in (400, 404)
        and "model"
        in (resp.text or "").lower()
    ):

        raise RuntimeError(
            f"model '{model}' not available "
            f"to this account "
            f"(HTTP {resp.status_code})"
        )

    resp.raise_for_status()

    data = resp.json()

    if (
        isinstance(data, dict)
        and data.get("error")
    ):

        err = data["error"]

        raise RuntimeError(
            str(
                err.get(
                    "message",
                    err,
                )
            )[:200]
        )

    choices = data.get("choices") or []

    if not choices:
        raise RuntimeError(
            "no choices returned"
        )

    content = (
        choices[0]
        .get("message", {})
        .get("content")
        or ""
    )

    usage = data.get("usage") or {}

    tin = (
        usage.get(
            "prompt_tokens",
            0,
        )
        or 0
    )

    tout = (
        usage.get(
            "completion_tokens",
            0,
        )
        or 0
    )

    log_cost(
        "mistral",
        task_type,
        {
            "input": tin,
            "output": tout,
        },
        0.0,
    )

    return {
        "content": content,
        "model_used": (
            data.get("model")
            or model
        ),
        "task_type": task_type,
        "tokens_used": {
            "input": tin,
            "output": tout,
        },
        "cost_usd": 0.0,
        "fell_back": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# GROQ
# ═══════════════════════════════════════════════════════════════════════════

def _call_groq(
    prompt,
    system,
    max_tokens,
    temperature,
    task_type,
) -> dict:

    model = groq_model()

    # IMPORTANT:
    # No hardcoded Groq model fallback.
    if not model:

        raise RuntimeError(
            "invalid model configuration: "
            "GROQ_MODEL is not set. "
            "Set the repository variable "
            "GROQ_MODEL to a model your "
            "Groq account can access."
        )

    wait_for_quota("groq")

    messages = []

    if system:

        messages.append(
            {
                "role": "system",
                "content": system,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    response = (
        _groq()
        .chat
        .completions
        .create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )

    if not response.choices:

        raise RuntimeError(
            "Groq returned no choices"
        )

    usage = response.usage

    content = (
        response
        .choices[0]
        .message
        .content
        or ""
    )

    log_cost(
        "groq",
        task_type,
        usage,
        0.0,
    )

    return {
        "content": content,
        "model_used": model,
        "task_type": task_type,
        "tokens_used": {
            "input": getattr(
                usage,
                "prompt_tokens",
                0,
            ),
            "output": getattr(
                usage,
                "completion_tokens",
                0,
            ),
        },
        "cost_usd": 0.0,
        "fell_back": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLAUDE PRICING
# ═══════════════════════════════════════════════════════════════════════════

_CLAUDE_PRICES = {
    "claude-sonnet-4-5": (
        3.0,
        15.0,
    ),
    "claude-sonnet-4-5-20250929": (
        3.0,
        15.0,
    ),
    "claude-haiku-4-5": (
        1.0,
        5.0,
    ),
    "claude-3-5-haiku": (
        0.80,
        4.0,
    ),
}


def _calculate_claude_cost(
    usage,
    model: str,
) -> float:

    rate_in = 3.0
    rate_out = 15.0

    for key, (
        ri,
        ro,
    ) in _CLAUDE_PRICES.items():

        if model.startswith(key):

            rate_in = ri
            rate_out = ro

            break

    return (
        usage.input_tokens
        * rate_in
        / 1_000_000
        +
        usage.output_tokens
        * rate_out
        / 1_000_000
    )


# ═══════════════════════════════════════════════════════════════════════════
# JSON HELPER
# ═══════════════════════════════════════════════════════════════════════════

def extract_json(
    text: str,
):

    if not text:
        return None

    fenced = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        re.DOTALL,
    )

    candidate = (
        fenced.group(1)
        if fenced
        else text
    )

    candidate = candidate.strip()

    try:

        return json.loads(candidate)

    except json.JSONDecodeError:
        pass

    for opener, closer in (
        ("{", "}"),
        ("[", "]"),
    ):

        for span in _iter_balanced_spans(
            candidate,
            opener,
            closer,
        ):

            try:

                return json.loads(span)

            except json.JSONDecodeError:

                continue

    return None


def _iter_balanced_spans(
    text: str,
    opener: str,
    closer: str,
):

    start = text.find(opener)

    while start != -1:

        depth = 0
        in_string = False
        escape = False

        for i in range(
            start,
            len(text),
        ):

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

                    yield text[
                        start:i + 1
                    ]

                    break

        start = text.find(
            opener,
            start + 1,
        )


# ═══════════════════════════════════════════════════════════════════════════
# PROVIDER CONNECTIVITY TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_providers() -> dict:
    """
    Ping configured providers with a tiny prompt.

    Only configured providers are tested.
    """

    results = {}

    probes = {
        "groq": lambda: _call_groq(
            "Reply with exactly: OK",
            None,
            16,
            0.0,
            "extract_text",
        ),

        "gemini": lambda: _call_gemini(
            "Reply with exactly: OK",
            None,
            16,
            0.0,
            "first_pass_filter",
        ),

        "openrouter": lambda: _call_openrouter(
            "Reply with exactly: OK",
            None,
            16,
            0.0,
            "extract_text",
        ),

        "mistral": lambda: _call_mistral(
            "Reply with exactly: OK",
            None,
            16,
            0.0,
            "extract_text",
        ),

        "claude": lambda: _call_claude(
            "Reply with exactly: OK",
            None,
            None,
            16,
            0.0,
            "final_scoring",
        ),
    }

    for name, fn in probes.items():

        if not provider_configured(name):

            results[name] = {
                "ok": False,
                "skipped": True,
                "reason": "not configured",
            }

            continue

        try:

            res = fn()

            results[name] = {
                "ok": True,
                "model": res["model_used"],
                "reply": (
                    res["content"]
                    or ""
                ).strip()[:40],
            }

        except Exception as e:

            results[name] = {
                "ok": False,
                "error": str(e),
            }

    return results
