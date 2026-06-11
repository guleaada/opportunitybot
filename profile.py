"""
profile.py — the user's hardcoded profile.

Everything the agent knows about *who it is searching for* lives here.
Edit the fields marked "I'll fill in" / "I'll confirm" before running.
"""

PROFILE = {
    "name": "Gulilat Kasiye Worku",
    "age": 27,
    "date_of_birth": "1999-04-27", 
    "nationality": "Ethiopian",
    "passport": "Ethiopian",
    "current_location": "Vientiane, Lao PDR",
    "current_status": "Working professional (NOT a student)",
    "education": {
        "degree": "BSc Psychology",
        "institution": "Injibara University, Ethiopia",
        "graduation_year": 2021,
        "gpa": 3.45,
    },
    "background": ["AI Developer", "Marketing Specialist", "Psychologist", "Community Leader"],
    "skills": [
        "Python", "Flask", "Claude AI API", "GitHub",
        "Digital Marketing", "AI/ML", "Web Development",
        'Solo founder, "AgriTech innovator"
    ],
    "languages": {
        "Amharic": "Native",
        "English": "Professional working proficiency (NO IELTS/TOEFL certificate)",
    },
    "english_certificate_alternatives_i_can_get": [
        "MOI (Medium of Instruction) certificate from Injibara University",
        "Duolingo English Test ($59, online, results in 48 hours)",
    ],
    "projects": [
        {
            "name": "SebilAI",
            "url": "https://sebilai.com",
            "description": "AI crop disease platform for Ethiopian smallholder farmers",
        },
        {"name": "JobsAI", "description": "AI career platform"},
    ],
    "preferred_destinations": [
        "Germany", "UK", "USA", "Canada", "Australia", "Japan",
        "Netherlands", "Sweden", "Norway", "Finland", "France", "Italy",
    ],
    "acceptable_destinations": [
        "South Korea", "Singapore", "China", "UAE", "Qatar",
        "New Zealand", "Ireland",
    ],
    "avoid_destinations": [],
    "preferred_program_types": [
        "Fully-funded masters scholarships",
        "Fellowships (no degree required)",
        "Residencies for AI/tech innovators",
        "Government scholarships (DAAD/Chevening/Fulbright/MEXT/etc)",
        "Tech accelerators with full sponsorship",
        "Research fellowships",
    ],
    "avoid_program_types": [
        "Fee-paying invitation summits (CSCD, CGDL, GBS, ICCSL, etc.)",
        "Conferences with no funding",
        "Programs requiring application fees over $50",
        "Programs that are 'partially funded' where I'd need to cover $5000+",
        "Online-only certificate programs",
    ],
    "email": "gulilatkasiye4@gmail.com",
    "whatsapp": "+251 704 161 402",
}


def profile_summary() -> str:
    """A compact, prose summary of the profile for stuffing into LLM prompts."""
    p = PROFILE
    edu = p["education"]
    return (
        f"Name: {p['name']} (age {p['age']}, {p['nationality']} national, "
        f"{p['passport']} passport).\n"
        f"Currently: {p['current_status']}, based in {p['current_location']}.\n"
        f"Education: {edu['degree']} from {edu['institution']} "
        f"(grad {edu['graduation_year']}).\n"
        f"Background: {', '.join(p['background'])}.\n"
        f"Skills: {', '.join(p['skills'])}.\n"
        f"Languages: {', '.join(f'{k} ({v})' for k, v in p['languages'].items())}.\n"
        f"English certificate: NONE yet, but can obtain "
        f"{', '.join(p['english_certificate_alternatives_i_can_get'])}.\n"
        f"Notable projects: "
        f"{', '.join(proj['name'] for proj in p['projects'])}.\n"
        f"Preferred destinations: {', '.join(p['preferred_destinations'])}.\n"
        f"Acceptable destinations: {', '.join(p['acceptable_destinations'])}.\n"
        f"Wants: {', '.join(p['preferred_program_types'])}.\n"
        f"Avoids: {', '.join(p['avoid_program_types'])}."
    )
