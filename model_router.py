"""
model_router.py

Central AI model router for OpportunityBot.

ROUTING
-------
High-stakes:
    Claude → scam detection / eligibility / final scoring
    If Claude is not configured or budget is exhausted:
        use the free provider chain.

Free provider chain:
    1. Groq
    2. OpenRouter
    3. Mistral
    4. Gemini

Provider order can be overridden with:

    MODEL_PROVIDER_ORDER=groq,openrouter,mistral,gemini

MODEL CONFIGURATION
-------------------
Models are configuration, not hardcoded provider choices.

    GROQ_MODEL
    OPENROUTER_MODEL
    MISTRAL_MODEL
    GEMINI_MODEL
    ANTHROPIC_MODEL
    ANTHROPIC_FALLBACK_MODEL

A provider that receives:
    429 → disabled for the rest of the scan
    404 / model-not-found → disabled for the rest of the scan

Other provider failures:
    → fallback to the next provider

Claude:
    → fail loudly for high-stakes work when Claude itself fails.

If all free providers fail:
    → Claude Haiku fallback, only when configured and budget allows.

All provider calls are logged with:
    provider
    model
    task
    token usage
    cost
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
# CLIENTS
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
                "ANTHROPIC_API_KEY is not configured"
            )

        _clients["anthropic"] = Anthropic(
            api_key=api_key
        )

    return _clients["anthropic"]


def _gemini():
    from google import genai

    if _clients["gemini"] is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured"
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
                "GROQ_API_KEY is not configured"
            )

        _clients["groq"] = Groq(
            api_key=api_key
        )

    return _clients["groq"]


# ═══════════════════════════════════════════════════════════════════════════
# TASK ROUTING
# ═══════════════════════════════════════════════════════════════════════════

TASK_ROUTING = {
    # HIGH STAKES
    "scam_detection": "claude",
    "deep_eligibility": "claude",
    "final_scoring": "claude",

    # NORMAL ANALYSIS
    "first_pass_filter": "gemini",
    "check_deadline": "gemini",
    "extract_document_requirements": "gemini",
    "estimate_complexity": "gemini",
    "generate_cover_letter": "gemini",
    "generate_email_subject": "gemini",
    "summarize_opportunity": "gemini",
    "classify_opportunity": "gemini",

    # MECHANICAL
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

SUPPORTED_FREE_PROVIDERS = set(
    DEFAULT_PROVIDER_ORDER
)


def provider_order() -> list:
    """
    Return the configured provider order.

    Invalid provider names are ignored.
    Duplicate providers are removed while preserving order.
    """

    raw = os.getenv(
        "MODEL_PROVIDER_ORDER",
        "",
    ).strip()

    if not raw:
        return list(DEFAULT_PROVIDER_ORDER)

    result = []

    for provider in raw.split(","):
        provider = provider.strip().lower()

        if (
            provider
            and provider in SUPPORTED_FREE_PROVIDERS
            and provider not in result
        ):
            result.append(provider)

    return result or list(DEFAULT_PROVIDER_ORDER)


# Backward compatibility.
FREE_CHAIN = list(DEFAULT_PROVIDER_ORDER)


# ═══════════════════════════════════════════════════════════════════════════
# ROUTING REASONS
# ═══════════════════════════════════════════════════════════════════════════

ROUTING_REASON = {
    "claude": "paid / high-stakes",
    "groq": "free/fallback tier",
    "openrouter": "independent fallback",
    "mistral": "free/fallback tier",
    "gemini": "free/final fallback",
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
# PROVIDER STATS / CIRCUIT BREAKER
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
    """
    Reset provider circuit-breaker state.

    Call this once at the beginning of every scan.
    """

    with _STATS_LOCK:
        _provider_stats.clear()


def provider_stats(name: str = None):
    with _STATS_LOCK:
        if name is None:
            return {
                key: dict(value)
                for key, value in _provider_stats.items()
            }

        return dict(
            _provider_stats.setdefault(
                name,
                _blank_stats(),
            )
        )


def _stats(name: str) -> dict:
    return _provider_stats.setdefault(
        name,
        _blank_stats(),
    )


def _record_success(provider: str) -> None:
    with _STATS_LOCK:
        stats = _stats(provider)

        stats["successful"] += 1
        stats["state"] = AVAILABLE


def _record_failure(
    provider: str,
    exc,
    task_type: str,
) -> None:

    short = _short_err(exc)

    with _STATS_LOCK:
        stats = _stats(provider)

        stats["last_error"] = short

        if _is_rate_limit(exc):
            stats["429"] += 1
            stats["state"] = RATE_LIMITED

            print(
                f"⚠️  {provider} {task_type}: "
                f"rate limited — disabling {provider} "
                f"for the rest of this scan"
            )

            return

        stats["failed"] += 1
        stats["state"] = FAILED

        if _is_model_not_found(exc):
            stats["404"] += 1
            stats["config_error"] = True

            print(
                f"⛔ {provider} {task_type}: "
                f"configured model unavailable — "
                f"{short}"
            )

            print(
                f"   ⛔ disabling {provider} "
                f"for the rest of this scan"
            )

        else:
            print(
                f"⚠️  {provider} {task_type}: "
                f"{short}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# PROVIDER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

def provider_configured(name: str) -> bool:

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

    with _STATS_LOCK:
        stats = _stats(name)

        if stats["state"] == RATE_LIMITED:
            return RATE_LIMITED

        if stats["config_error"]:
            return FAILED

        if stats["successful"] > 0:
            return AVAILABLE

        if stats["state"] == FAILED:
            return FAILED

        return ENABLED


# ═══════════════════════════════════════════════════════════════════════════
# ERROR CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def _status_code(exc) -> Optional[int]:

    response = getattr(
        exc,
        "response",
        None,
    )

    status = getattr(
        response,
        "status_code",
        None,
    )

    if status is not None:
        return status

    return getattr(
        exc,
        "status_code",
        None,
    )


def _is_rate_limit(exc) -> bool:

    status = _status_code(exc)

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
    "model does not exist",
    "does not exist",
    "invalid model",
    "invalid_model",
    "unknown model",
    "unsupported model",
    "model is not available",
    "model unavailable",
    "not a valid model",
    "decommissioned",
    "retired model",
    "model retired",
)


def _is_model_not_found(exc) -> bool:

    status = _status_code(exc)

    if status == 404:
        return True

    text = str(exc).lower()

    if any(
        marker in text
        for marker in _MODEL_CONFIG_ERRORS
    ):
        return True

    return (
        "model" in text
        and (
            "not found" in text
            or "404" in text
        )
    )


def _short_err(exc) -> str:

    status = _status_code(exc)

    message = " ".join(
        str(exc).split()
    )

    message = message[:140]

    if status:
        return f"{status} {message}"[:160]

    return message


# ═══════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

def groq_model() -> str:
    """
    Groq model MUST come from GROQ_MODEL.

    No hardcoded Groq model is used.
    """

    return os.getenv(
        "GROQ_MODEL",
        "",
    ).strip()


def openrouter_model() -> str:
    """
    OpenRouter model.

    OPENROUTER_MODEL is configurable.
    """

    return os.getenv(
        "OPENROUTER_MODEL",
        "openrouter/free",
    ).strip()


def mistral_model() -> str:
    """
    Mistral model.

    MISTRAL_MODEL is configurable.
    """

    return os.getenv(
        "MISTRAL_MODEL",
        "mistral-small-latest",
    ).strip()


def gemini_model() -> str:
    """
    Gemini model.

    GEMINI_MODEL is configurable.
    """

    return os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    ).strip()


def claude_model() -> str:
    """
    Primary Claude model.

    ANTHROPIC_MODEL is configurable.
    """

    return os.getenv(
        "ANTHROPIC_MODEL",
        "claude-sonnet-4-5",
    ).strip()


def claude_fallback_model() -> str:
    """
    Claude fallback model.

    ANTHROPIC_FALLBACK_MODEL is configurable.
    """

    return os.getenv(
        "ANTHROPIC_FALLBACK_MODEL",
        "claude-haiku-4-5",
    ).strip()


# ═══════════════════════════════════════════════════════════════════════════
# CLAUDE BUDGET
# ═══════════════════════════════════════════════════════════════════════════

def claude_budget_caps():

    daily = float(
        os.getenv(
            "DAILY_CLAUDE_BUDGET_USD",
            "0.50",
        )
    )

    monthly = float(
        os.getenv(
            "MONTHLY_CLAUDE_BUDGET_USD",
            "10.00",
        )
    )

    return daily, monthly


def has_claude_budget() -> bool:

    try:
        daily_cap, monthly_cap = (
            claude_budget_caps()
        )

        daily_spend = (
            get_daily_claude_spend()
        )

        monthly_spend = (
            get_monthly_claude_spend()
        )

        return (
            daily_spend < daily_cap
            and monthly_spend < monthly_cap
        )

    except Exception as exc:

        print(
            f"⚠️  Could not read Claude budget "
            f"({exc}) — assuming exhausted"
        )

        return False


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

    with _STATS_LOCK:
        stats = _stats(provider)

        return (
            stats["state"] != RATE_LIMITED
            and not stats["config_error"]
        )


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

    # High-stakes tasks prefer Claude.
    if (
        primary == "claude"
        and _provider_available("claude")
    ):

        print(
            f"🤖 [{task_type}] → claude "
            f"({ROUTING_REASON['claude']})"
        )

        try:
            return _call_claude(
                prompt,
                system,
                tools,
                max_tokens,
                temperature,
                task_type,
            )

        except Exception as exc:

            print(
                f"⛔ Claude failed for "
                f"{task_type}: "
                f"{_short_err(exc)}"
            )

            raise

    # All other work, and high-stakes work without Claude,
    # goes through the free chain.
    return _call_free_chain(
        prompt,
        system,
        max_tokens,
        temperature,
        task_type,
    )


# ═══════════════════════════════════════════════════════════════════════════
# FREE PROVIDER CHAIN
# ═══════════════════════════════════════════════════════════════════════════

class ProvidersUnavailable(RuntimeError):
    """
    Every configured free provider was unavailable.

    This is a technical failure.

    It must NEVER be interpreted as:
        INELIGIBLE
        SUSPICIOUS
        REJECTED

    Callers should preserve the candidate and mark analysis
    as unavailable.
    """


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

    # Show providers already unavailable.
    for provider in order:

        if provider not in chain:

            print(
                f"  ↷ [{task_type}] "
                f"skipping {provider} "
                f"({provider_state(provider)})"
            )

    attempted = []
    throttled = []

    for index, provider in enumerate(chain):

        if not _provider_available(provider):
            continue

        wait = _throttle_wait(provider)

        if wait > MAX_PROVIDER_WAIT_SECONDS:

            with _STATS_LOCK:
                _stats(provider)[
                    "soft_skips"
                ] += 1

            throttled.append(
                (
                    wait,
                    provider,
                )
            )

            next_provider = (
                chain[index + 1]
                if index + 1 < len(chain)
                else "nothing left"
            )

            print(
                f"  ⏭️  [{task_type}] "
                f"{provider} needs "
                f"{_wait_str(wait)} "
                f"of rate-limit sleep — "
                f"advancing to "
                f"{next_provider}"
            )

            continue

        result = _attempt(
            provider,
            index,
            prompt,
            system,
            max_tokens,
            temperature,
            task_type,
        )

        if result is not None:
            return result

        attempted.append(provider)

        next_provider = (
            chain[index + 1]
            if index + 1 < len(chain)
            else "nothing left"
        )

        print(
            f"   → falling back to "
            f"{next_provider}"
        )

    # If every provider was throttled, wait for the
    # soonest one rather than immediately failing.
    if not attempted and throttled:

        wait, provider = min(
            throttled,
            key=lambda item: item[0],
        )

        if wait != float("inf"):

            print(
                f"⏳ [{task_type}] "
                f"all providers are rate-limited — "
                f"waiting {_wait_str(wait)} "
                f"for {provider}"
            )

            result = _attempt(
                provider,
                chain.index(provider),
                prompt,
                system,
                max_tokens,
                temperature,
                task_type,
            )

            if result is not None:
                return result

            attempted.append(provider)

    # Optional final Claude fallback.
    #
    # This is ONLY used if:
    #   - Claude is configured
    #   - Claude budget allows it
    #
    # It is deliberately NOT used as an automatic
    # replacement for high-stakes Claude failures.
    if (
        provider_configured("claude")
        and has_claude_budget()
    ):

        fallback_model = (
            claude_fallback_model()
        )

        if not fallback_model:
            print(
                f"⚠️  [{task_type}] "
                f"Claude fallback is configured "
                f"but ANTHROPIC_FALLBACK_MODEL is empty"
            )
        else:
            print(
                f"🆘 [{task_type}] "
                f"all free providers failed — "
                f"using Claude fallback "
                f"{fallback_model}"
            )

            try:
                return _call_claude(
                    prompt,
                    system,
                    None,
                    max_tokens,
                    temperature,
                    task_type,
                    model_override=fallback_model,
                )

            except Exception as exc:

                print(
                    f"⛔ Claude fallback failed: "
                    f"{_short_err(exc)}"
                )

    raise ProvidersUnavailable(
        "all configured providers unavailable "
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
        return float(
            seconds_until_available(
                provider
            )
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
        _stats(provider)[
            "requests"
        ] += 1

    print(
        f"🤖 [{task_type}] → "
        f"{provider} "
        f"({ROUTING_REASON.get(provider, 'free')})"
    )

    try:

        result = _dispatch_free(
            provider,
            prompt,
            system,
            max_tokens,
            temperature,
            task_type,
        )

    except Exception as exc:

        _record_failure(
            provider,
            exc,
            task_type,
        )

        return None

    _record_success(provider)

    result["fell_back"] = (
        index > 0
    )

    return result


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
):

    wait_for_quota(
        "claude"
    )

    model = (
        model_override
        or claude_model()
    )

    if not model:
        raise RuntimeError(
            "ANTHROPIC_MODEL is not configured"
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

    response = (
        _anthropic()
        .messages
        .create(**kwargs)
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
            "input": (
                response
                .usage
                .input_tokens
            ),
            "output": (
                response
                .usage
                .output_tokens
            ),
        },
        "cost_usd": cost,
        "fell_back": bool(
            model_override
        ),
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
):

    wait_for_quota(
        "gemini"
    )

    full_prompt = (
        f"{system}\n\n{prompt}"
        if system
        else prompt
    )

    model = gemini_model()

    if not model:
        raise RuntimeError(
            "GEMINI_MODEL is empty"
        )

    response = (
        _gemini()
        .models
        .generate_content(
            model=model,
            contents=full_prompt,
            config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
    )

    input_tokens = 0
    output_tokens = 0

    usage = getattr(
        response,
        "usage_metadata",
        None,
    )

    if usage is not None:

        input_tokens = (
            getattr(
                usage,
                "prompt_token_count",
                0,
            )
            or 0
        )

        output_tokens = (
            getattr(
                usage,
                "candidates_token_count",
                0,
            )
            or 0
        )

    log_cost(
        "gemini",
        task_type,
        {
            "input": input_tokens,
            "output": output_tokens,
        },
        0.0,
    )

    return {
        "content": _gemini_text(
            response
        ),
        "model_used": model,
        "task_type": task_type,
        "tokens_used": {
            "input": input_tokens,
            "output": output_tokens,
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
):

    api_key = os.getenv(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set"
        )

    model = openrouter_model()

    if not model:
        raise RuntimeError(
            "OPENROUTER_MODEL is empty"
        )

    wait_for_quota(
        "openrouter"
    )

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

    response = requests.post(
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

    if response.status_code == 429:
        raise RuntimeError(
            "429 rate limited"
        )

    if response.status_code == 404:
        raise RuntimeError(
            f"404 model not found: {model}"
        )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):

        error = data["error"]

        raise RuntimeError(
            f"{error.get('code', 'error')}: "
            f"{error.get('message', error)}"
        )

    choices = (
        data.get("choices")
        or []
    )

    if not choices:
        raise RuntimeError(
            "OpenRouter returned no choices"
        )

    content = (
        choices[0]
        .get("message", {})
        .get("content")
        or ""
    )

    usage = (
        data.get("usage")
        or {}
    )

    input_tokens = (
        usage.get(
            "prompt_tokens",
            0,
        )
        or 0
    )

    output_tokens = (
        usage.get(
            "completion_tokens",
            0,
        )
        or 0
    )

    actual_model = (
        data.get("model")
        or model
    )

    log_cost(
        "openrouter",
        task_type,
        {
            "input": input_tokens,
            "output": output_tokens,
        },
        0.0,
    )

    return {
        "content": content,
        "model_used": actual_model,
        "task_type": task_type,
        "tokens_used": {
            "input": input_tokens,
            "output": output_tokens,
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
):

    api_key = os.getenv(
        "MISTRAL_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY not set"
        )

    model = mistral_model()

    if not model:
        raise RuntimeError(
            "MISTRAL_MODEL is empty"
        )

    wait_for_quota(
        "mistral"
    )

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

    response = requests.post(
        MISTRAL_URL,
        headers={
            "Authorization": (
                f"Bearer {api_key}"
            ),
            "Content-Type": (
                "application/json"
            ),
            "Accept": (
                "application/json"
            ),
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

    if response.status_code == 429:
        raise RuntimeError(
            "429 rate limited"
        )

    if response.status_code == 404:
        raise RuntimeError(
            f"404 model not found: {model}"
        )

    if (
        response.status_code == 400
        and "model" in (
            response.text or ""
        ).lower()
    ):
        raise RuntimeError(
            f"model '{model}' unavailable "
            f"to this account"
        )

    response.raise_for_status()

    data = response.json()

    if data.get("error"):

        error = data["error"]

        raise RuntimeError(
            str(
                error.get(
                    "message",
                    error,
                )
            )[:200]
        )

    choices = (
        data.get("choices")
        or []
    )

    if not choices:
        raise RuntimeError(
            "Mistral returned no choices"
        )

    content = (
        choices[0]
        .get("message", {})
        .get("content")
        or ""
    )

    usage = (
        data.get("usage")
        or {}
    )

    input_tokens = (
        usage.get(
            "prompt_tokens",
            0,
        )
        or 0
    )

    output_tokens = (
        usage.get(
            "completion_tokens",
            0,
        )
        or 0
    )

    actual_model = (
        data.get("model")
        or model
    )

    log_cost(
        "mistral",
        task_type,
        {
            "input": input_tokens,
            "output": output_tokens,
        },
        0.0,
    )

    return {
        "content": content,
        "model_used": actual_model,
        "task_type": task_type,
        "tokens_used": {
            "input": input_tokens,
            "output": output_tokens,
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
):

    model = groq_model()

    if not model:
        raise RuntimeError(
            "GROQ_MODEL is not configured. "
            "Set the repository variable GROQ_MODEL."
        )

    wait_for_quota(
        "groq"
    )

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

    usage = response.usage

    content = (
        response
        .choices[0]
        .message
        .content
        or ""
    )

    input_tokens = getattr(
        usage,
        "prompt_tokens",
        0,
    )

    output_tokens = getattr(
        usage,
        "completion_tokens",
        0,
    )

    log_cost(
        "groq",
        task_type,
        usage,
        0.0,
    )

    return {
        "finish_reason": getattr(response.choices[0], "finish_reason", None),
        "content": content,
        "model_used": model,
        "task_type": task_type,
        "tokens_used": {
            "input": input_tokens,
            "output": output_tokens,
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

    input_rate = 3.0
    output_rate = 15.0

    for key, (
        input_price,
        output_price,
    ) in _CLAUDE_PRICES.items():

        if model.startswith(key):

            input_rate = input_price
            output_rate = output_price

            break

    return (
        usage.input_tokens
        * input_rate
        / 1_000_000
        +
        usage.output_tokens
        * output_rate
        / 1_000_000
    )


# ═══════════════════════════════════════════════════════════════════════════
# MODEL HEALTH
# ═══════════════════════════════════════════════════════════════════════════

def model_health_lines() -> list:

    lines = [
        "🧠 MODEL HEALTH"
    ]

    names = list(
        provider_order()
    )

    if "claude" not in names:
        names.append(
            "claude"
        )

    for name in names:

        configured = (
            provider_configured(name)
        )

        stats = provider_stats(
            name
        )

        state = provider_state(
            name
        )

        if not configured:

            lines.append(
                f"   {name.capitalize():11} "
                f"configured: NO   "
                f"status: {DISABLED}"
            )

            continue

        detail = (
            f"requests {stats['requests']}, "
            f"ok {stats['successful']}, "
            f"429 {stats['429']}, "
            f"404 {stats['404']}, "
            f"failed {stats['failed']}"
        )

        if stats["soft_skips"]:
            detail += (
                f", throttle-skipped "
                f"{stats['soft_skips']}"
            )

        lines.append(
            f"   {name.capitalize():11} "
            f"configured: YES  "
            f"status: {state}  •  "
            f"{detail}"
        )

        if name == "groq":
            lines.append(
                f"               model: "
                f"{groq_model() or 'NOT SET'}"
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
                f"{claude_model()}"
            )

        if stats["last_error"]:
            lines.append(
                f"               last error: "
                f"{stats['last_error'][:90]}"
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

    for provider in order:

        if not provider_configured(
            provider
        ):

            print(
                f"   {provider:11} "
                f"{DISABLED} "
                f"(no API key configured)"
            )

            continue

        if provider == "groq":

            print(
                f"   {provider:11} READY  "
                f"model: "
                f"{groq_model() or 'NOT SET'}"
            )

        elif provider == "openrouter":

            print(
                f"   {provider:11} READY  "
                f"model: "
                f"{openrouter_model()}"
            )

        elif provider == "mistral":

            print(
                f"   {provider:11} READY  "
                f"model: "
                f"{mistral_model()}"
            )

        elif provider == "gemini":

            print(
                f"   {provider:11} READY  "
                f"model: "
                f"{gemini_model()}"
            )

    if not provider_configured(
        "claude"
    ):

        print(
            f"   {'claude':11} "
            f"{DISABLED} "
            f"(no API key configured)"
        )

    else:

        print(
            f"   {'claude':11} READY  "
            f"model: {claude_model()}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# JSON EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def as_object(data, where: str) -> dict:
    """A decoded model reply, but only when it is actually a JSON object.

    extract_json() returns the first balanced ``{...}`` OR ``[...]`` it finds,
    so a model that answers with an array hands back a list — and a list has
    no .get(), which is how a scan died with
    ``AttributeError: 'list' object has no attribute 'get'``.

    Anything that is not a dict carries none of the fields we asked for.
    Unwrapping a one-element array or reading positionally would be inventing
    a judgement the model did not make, so the reply is reported and
    discarded, leaving the caller to fall back to its own safe default.
    """
    if isinstance(data, dict):
        return data
    if data is not None:
        print(f"⚠️  {where}: model returned a JSON {type(data).__name__}, "
              f"expected an object — treating the response as unusable.")
    return {}


def extract_json(
    text: str,
):
    """
    Best-effort JSON extraction.

    Handles:
        raw JSON
        ```json fences
        JSON surrounded by prose
    """

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
        return json.loads(
            candidate
        )

    except json.JSONDecodeError:
        pass

    # Decode only the first outer JSON value. Never salvage a nested object
    # from a truncated envelope (e.g. breakdown masquerading as a score).
    starts = [i for i in (candidate.find("{"), candidate.find("[")) if i >= 0]
    if not starts:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate[min(starts):])
        return value
    except json.JSONDecodeError:
        return None


def _iter_balanced_spans(
    text: str,
    opener: str,
    closer: str,
):

    start = text.find(
        opener
    )

    while start != -1:

        depth = 0
        in_string = False
        escape = False

        for index in range(
            start,
            len(text),
        ):

            char = text[index]

            if in_string:

                if escape:
                    escape = False

                elif char == "\\":
                    escape = True

                elif char == '"':
                    in_string = False

                continue

            if char == '"':
                in_string = True

            elif char == opener:
                depth += 1

            elif char == closer:

                depth -= 1

                if depth == 0:

                    yield text[
                        start:index + 1
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
    Test every configured provider.

    Uses the same configured model that the real scan uses.

    Does NOT test providers that have no API key.
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

        "gemini": lambda: _call_gemini(
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

    for provider in (
        provider_order()
        + ["claude"]
    ):

        if provider in results:
            continue

        if not provider_configured(
            provider
        ):

            results[provider] = {
                "ok": False,
                "skipped": True,
                "reason": (
                    "API key not configured"
                ),
            }

            continue

        fn = probes.get(
            provider
        )

        if fn is None:
            continue

        try:

            result = fn()

            results[provider] = {
                "ok": True,
                "model": result[
                    "model_used"
                ],
                "reply": (
                    result["content"]
                    or ""
                ).strip()[:40],
            }

        except Exception as exc:

            results[provider] = {
                "ok": False,
                "error": str(exc),
            }

    return results
