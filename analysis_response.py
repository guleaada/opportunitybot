"""Validate decision responses; technical failures must remain retryable."""
import math
from model_router import extract_json


class AnalysisResponseError(RuntimeError):
    pass


def decision_response(call, task, prompt, system, max_tokens, validate):
    for attempt in range(2):
        limit = max_tokens * (attempt + 1)
        res = call(task, prompt, system=system, max_tokens=limit)
        parsed = extract_json(res.get("content", ""))
        # Token counts are a conservative fallback for adapters that do not
        # expose finish_reason. A response at the limit may be incomplete.
        truncated = (res.get("finish_reason") in ("length", "max_tokens") or
                     (res.get("tokens_used") or {}).get("output", 0) >= limit)
        if not truncated and isinstance(parsed, dict) and validate(parsed):
            return res, parsed
    raise AnalysisResponseError(f"{task}: incomplete or invalid response after retry")


def valid_score(data):
    score = data.get("overall_score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return False
    if not math.isfinite(score) or not 0 <= score <= 10:
        return False
    if data.get("funding", "unknown") not in ("fully_funded", "partial", "none", "unknown"):
        return False
    return True


def valid_eligibility(data):
    return (data.get("eligibility_status") in (
        "CONFIRMED_ELIGIBLE", "PROBABLY_ELIGIBLE", "UNCERTAIN",
        "PROBABLY_INELIGIBLE", "CONFIRMED_INELIGIBLE") or
        data.get("overall") in ("eligible", "probably_eligible", "uncertain", "ineligible", "unknown"))
