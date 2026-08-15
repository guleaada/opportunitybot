"""
model_router.py — routes each task to the correct AI model based on stakes.

PHILOSOPHY
----------
- Claude Sonnet (paid)  → high-stakes judgment ONLY (scam, eligibility, scoring)
- Gemini Flash (free)   → medium tasks (filters, extraction, drafts)
- Groq Llama (free)     → mechanical tasks (clean text, translate)

Plus:
- Hard daily/monthly budget cap on Claude (downgrade to Gemini if exceeded).
- Graceful fallback chain when free models fail.
- Every call is logged with which model ran and what it cost.

Fallback chain (per spec section D):
    Claude fails           → re-raise (fail loud; high stakes)
    Gemini fails           → Groq
    Groq fails             → Gemini
    ALL free models fail   → Claude Haiku 4.5 (last resort)
"""

import json
import os
import re
from typing import Optional

from cost_tracker import log_cost, get_daily_claude_spend, get_monthly_claude_spend
from rate_limiter import wait_for_quota, DailyQuotaExceeded

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
    # MECHANICAL — Groq (free)
    "clean_html": "groq",
    "extract_text": "groq",
    "translate": "groq",
}

# Human-readable "why this model" used for the transparency log.
ROUTING_REASON = {
    "claude": "high-stakes judgment",
    "gemini": "free / medium task",
    "groq": "free / mechanical task",
}


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

    # ── Budget guardrail: downgrade Claude → Gemini if caps are hit ───────
    if primary == "claude":
        daily = get_daily_claude_spend()
        monthly = get_monthly_claude_spend()
        daily_cap, monthly_cap = claude_budget_caps()
        if daily >= daily_cap:
            print(f"⚠️  Claude daily budget ${daily_cap} reached "
                  f"(spent ${daily:.3f}). Downgrading '{task_type}' → Gemini.")
            primary = "gemini"
        elif monthly >= monthly_cap:
            print(f"⚠️  Claude monthly budget ${monthly_cap} reached "
                  f"(spent ${monthly:.2f}). Downgrading '{task_type}' → Gemini.")
            primary = "gemini"

    print(f"🤖 [{task_type}] → {primary} ({ROUTING_REASON.get(primary, '?')})")

    try:
        if primary == "claude":
            return _call_claude(prompt, system, tools, max_tokens, temperature, task_type)
        if primary == "gemini":
            return _call_gemini(prompt, system, max_tokens, temperature, task_type)
        if primary == "groq":
            return _call_groq(prompt, system, max_tokens, temperature, task_type)
    except Exception as e:
        print(f"⚠️  {primary} failed for {task_type}: {e}")
        if primary == "claude":
            raise  # High-stakes; fail loud (no silent downgrade on error).
        if primary == "gemini":
            return _try_fallback("groq", "gemini", prompt, system, max_tokens,
                                 temperature, task_type)
        if primary == "groq":
            return _try_fallback("gemini", "groq", prompt, system, max_tokens,
                                 temperature, task_type)
    raise RuntimeError(f"Unroutable task_type: {task_type}")


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


def _try_fallback(fallback, failed, prompt, system, max_tokens, temperature, task_type):
    """Try the sibling free model.

    If that also fails, a paid last-resort is only allowed when there is
    genuine Claude budget headroom. Bulk free-tier work (clean_html,
    first_pass_filter, classify_opportunity, ...) must never quietly escalate
    to a paid model just because both free providers were down — that would
    bypass the budget guardrail, which only runs for Claude-primary tasks.
    """
    print(f"  → Falling back from {failed} to {fallback}")
    try:
        if fallback == "groq":
            res = _call_groq(prompt, system, max_tokens, temperature, task_type)
        else:
            res = _call_gemini(prompt, system, max_tokens, temperature, task_type)
        res["fell_back"] = True
        return res
    except Exception as e2:
        print(f"⚠️  Fallback {fallback} also failed: {e2}")
        if not has_claude_budget():
            daily_cap, monthly_cap = claude_budget_caps()
            print(f"  ⛔ No Claude budget (daily cap ${daily_cap}, monthly "
                  f"${monthly_cap}) — refusing to escalate free task "
                  f"'{task_type}' to a paid model.")
            raise RuntimeError(
                f"both free providers failed for '{task_type}' and the paid "
                f"fallback is blocked by the Claude budget") from e2
        print("  → Last resort: Claude Haiku 4.5")
        res = _call_claude(prompt, system, None, max_tokens, temperature,
                           task_type, model_override=os.getenv(
                               "ANTHROPIC_FALLBACK_MODEL", "claude-haiku-4-5"))
        res["fell_back"] = True
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
