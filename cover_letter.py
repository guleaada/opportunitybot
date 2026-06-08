"""
cover_letter.py — draft cover-letter / motivation-letter generator (Gemini).

Creative + revisable, so it routes to the free Gemini model
(task_type="generate_cover_letter"). Output is a DRAFT for the user to edit,
never sent automatically.
"""

from model_router import call_model
from profile import PROFILE, profile_summary


def generate_cover_letter(data: dict, profile: dict = None) -> dict:
    """Generate a motivation-letter draft for an opportunity.

    ``data`` should include at least ``title``/``name`` and ``raw_text`` (or a
    summary). Returns {"draft": str, "model_used": str}.
    """
    profile = profile or PROFILE
    name = data.get("title") or data.get("name") or "this opportunity"
    context = data.get("raw_text") or data.get("snippet") or ""
    docs = data.get("documents") or {}

    system = (
        "You are an expert scholarship/fellowship application writer. You write "
        "authentic, specific, non-generic motivation letters. Avoid clichés and "
        "empty superlatives. Ground claims in the candidate's real projects."
    )
    prompt = (
        f"Write a ~400-word motivation letter draft for the candidate applying "
        f"to: {name}.\n\n"
        f"CANDIDATE PROFILE:\n{profile_summary()}\n\n"
        f"Emphasize the candidate's real AI projects (SebilAI — AI crop disease "
        f"platform for Ethiopian farmers; JobsAI — AI career platform), their "
        f"psychology + community-leadership background, and motivation to use "
        f"the program to scale impact in Ethiopia/Africa.\n\n"
        f"If relevant document requirements were detected, weave them in "
        f"naturally: {docs}\n\n"
        f"OPPORTUNITY CONTEXT (for tailoring):\n{context[:4000]}\n\n"
        "Return only the letter text (no preamble, no markdown headers). "
        "Use placeholders like [Program Name] / [Date] only where you truly "
        "lack the information."
    )
    res = call_model("generate_cover_letter", prompt, system=system,
                     max_tokens=900, temperature=0.6)
    return {"draft": (res["content"] or "").strip(), "model_used": res["model_used"]}
