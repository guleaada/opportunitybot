#!/usr/bin/env python3
"""GROQ_MODEL is configuration, with no hardcoded fallback.

Groq decommissioned llama-3.3-70b-versatile on 2026-08-16 and every call
404'd two days later. A hardcoded default only postpones the next outage, so
the model id is read from config and an unset value fails loudly instead of
pretending to be a working default.

Run:  python tests/test_groq_model_config.py
"""
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import model_router as mr

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"  ✅ {msg}")


DEAD = ("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-",
        "8b-instant", "70b-versatile")

# A retired id may still be named in prose — the comment explaining WHY there
# is no default is worth keeping. What must never come back is a retired id
# used as a VALUE: a getenv fallback, an `or` default, a workflow `||`
# fallback, or an assignment in .env.example.
VALUE_PATTERNS = [
    r"getenv\([^)]*,\s*['\"]{dead}",     # os.getenv("X", "<dead>")
    r"\bor\s+['\"]{dead}",               # os.getenv("X") or "<dead>"
    r"\|\|\s*['\"]{dead}",              # ${{ ... || '<dead>' }}
    r"=\s*{dead}",                        # KEY=<dead> in .env.example
]

# ══════════════════════════════════════════════════════════════════════════
print("\n1. No retired model id is used as a value anywhere")
offenders = []
for path in ROOT.rglob("*"):
    if not path.is_file() or path.suffix not in {".py", ".yml", ".yaml",
                                                 ".md", ".json", ".example"}:
        continue
    rel = path.relative_to(ROOT)
    if rel.parts[0] in {".git", "data", "__pycache__", "drafts", ".cache"}:
        continue
    if rel == Path("tests/test_groq_model_config.py"):
        continue                      # this guard names the ids on purpose
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    for token in DEAD:
        for pat in VALUE_PATTERNS:
            if re.search(pat.format(dead=re.escape(token)), text):
                offenders.append(f"{rel}: {token} used as a value")
assert not offenders, "retired model ids used as values:\n  " + "\n  ".join(offenders)
ok(f"no retired id ({', '.join(DEAD)}) is a getenv/or/|| default or an "
   f"env assignment")

# ══════════════════════════════════════════════════════════════════════════
print("\n2. groq_model() is a pure config read")
with patch.dict(os.environ, {"GROQ_MODEL": "openai/gpt-oss-120b"}):
    assert mr.groq_model() == "openai/gpt-oss-120b"
with patch.dict(os.environ, {"GROQ_MODEL": "  padded/model  "}):
    assert mr.groq_model() == "padded/model", "value should be stripped"
with patch.dict(os.environ, {"GROQ_MODEL": ""}):
    assert mr.groq_model() == "", "empty must NOT resolve to a default"
env = dict(os.environ)
env.pop("GROQ_MODEL", None)
with patch.dict(os.environ, env, clear=True):
    assert mr.groq_model() == "", "unset must NOT resolve to a default"
ok("set / padded / empty / unset all handled, and no default is invented")

# ══════════════════════════════════════════════════════════════════════════
print("\n3. An unset GROQ_MODEL fails loudly — and latches, not silently")
for value in ("", "   "):
    with patch.dict(os.environ, {"GROQ_MODEL": value}), \
         patch.object(mr, "wait_for_quota") as waited, \
         patch.object(mr, "_groq") as client:
        try:
            mr._call_groq("p", None, 10, 0.0, "first_pass_filter")
            raise AssertionError("an unset GROQ_MODEL must raise")
        except RuntimeError as e:
            msg = str(e)
        assert "GROQ_MODEL" in msg, msg
        assert waited.call_count == 0, "should fail before spending quota"
        assert client.call_count == 0, "should fail before building a client"
ok("raises before touching the quota window or the SDK, and names the variable")

# The message must be classified as a CONFIG error, so the existing latch
# disables groq for the scan instead of retrying it once per candidate.
err = RuntimeError("invalid model configuration: GROQ_MODEL is not set.")
assert mr._is_model_not_found(err) is True, "must latch like a 404"
assert mr._is_rate_limit(err) is False, "must not look like a rate limit"
ok("classified as a configuration error → latches for the scan, chain advances")

# ══════════════════════════════════════════════════════════════════════════
print("\n4. The configured model is what actually gets sent")
with patch.dict(os.environ, {"GROQ_MODEL": "openai/gpt-oss-120b"}), \
     patch.object(mr, "wait_for_quota"), \
     patch.object(mr, "log_cost"), \
     patch.object(mr, "_groq") as client:
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content="hello"))]
    completion.usage = MagicMock(prompt_tokens=5, completion_tokens=7)
    client.return_value.chat.completions.create.return_value = completion
    out = mr._call_groq("p", "sys", 100, 0.3, "first_pass_filter")

sent = client.return_value.chat.completions.create.call_args.kwargs
assert sent["model"] == "openai/gpt-oss-120b", sent["model"]
assert out["model_used"] == "openai/gpt-oss-120b", out["model_used"]
assert out["cost_usd"] == 0.0
ok("the env value reaches the API call and the reported model_used")

# ══════════════════════════════════════════════════════════════════════════
print("\n5. Logs surface the effective model, never a stale guess")
with patch.dict(os.environ, {"GROQ_MODEL": "openai/gpt-oss-120b",
                             "GROQ_API_KEY": "k"}):
    health = "\n".join(mr.model_health_lines())
    assert "openai/gpt-oss-120b" in health, health
with patch.dict(os.environ, {"GROQ_MODEL": "", "GROQ_API_KEY": "k"}):
    health = "\n".join(mr.model_health_lines())
    assert "NOT SET" in health, health
ok("MODEL HEALTH prints the configured id, or 'NOT SET' when it is missing")

# ══════════════════════════════════════════════════════════════════════════
print("\n6. The workflow reads model ids from Variables, with no literal")
wf = (ROOT / ".github/workflows/daily_scan.yml").read_text(encoding="utf-8")
for var in ("GROQ_MODEL", "GEMINI_MODEL"):
    line = next(l for l in wf.splitlines()
                if re.match(rf"\s*{var}:", l))
    assert f"vars.{var}" in line, f"{var} should read a repo Variable: {line}"
    assert f"secrets.{var}" not in line, f"{var} still reads a secret: {line}"
    assert "||" not in line, f"{var} still has a hardcoded fallback: {line}"
ok("GROQ_MODEL and GEMINI_MODEL both read vars.*, neither has a literal")

# Model ids as Variables are visible in logs; API keys stay secrets.
for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "MISTRAL_API_KEY",
            "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"):
    line = next((l for l in wf.splitlines() if re.match(rf"\s*{key}:", l)), None)
    if line:
        assert f"secrets.{key}" in line, f"{key} must stay a secret: {line}"
ok("every API key still reads secrets.* — only model ids moved")

print(f"\n{'=' * 62}\n✅ ALL {len(PASSED)} CHECKS PASSED\n{'=' * 62}")
