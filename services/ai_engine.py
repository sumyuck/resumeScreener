"""
AI engine: LLM-powered extraction, scoring, quality checks, and interview question generation.
Uses Groq API via OpenAI-compatible client.
"""

import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"


def _get_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY must be set in .env")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def _chat(prompt: str, temperature: float = 0.3) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()


def _parse_json_response(text: str) -> dict | list:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}


def extract_fields(resume_text: str, extraction_config: list[dict]) -> dict:
    valid_keys = {field["key"] for field in extraction_config}

    field_descriptions = []
    for field in extraction_config:
        req = "required" if field.get("required") else "optional"
        field_descriptions.append(
            f'- "{field["key"]}" ({field["label"]}, type: {field["type"]}, {req})'
        )
    fields_str = "\n".join(field_descriptions)

    prompt = f"""You are a precise resume information extractor.

Extract ONLY the following fields from the resume text below. Return ONLY valid JSON.
Do NOT include any fields not listed below.

Fields to extract:
{fields_str}

Rules:
1. For "list" type fields, return a JSON array of strings.
2. For "number" type fields, return a numeric value or null.
3. For "text" type fields, return a string or null.
4. If a field cannot be found, return null for that field.
5. For "years_of_experience", calculate from work history if not explicitly stated.
6. Be precise. Do not invent or hallucinate information.
7. Return ONLY the fields listed above. No extra fields.

Resume text:
\"\"\"
{resume_text[:6000]}
\"\"\"

Return ONLY a JSON object with the field keys as keys."""

    response = _chat(prompt)
    result = _parse_json_response(response)
    # Filter to only include configured field keys
    return {k: v for k, v in result.items() if k in valid_keys}


_MATCH_LEVEL_SCORES = {
    "strong_match": 9,
    "moderate_match": 6.5,
    "weak_match": 4,
    "no_match": 1.5,
}

_CATEGORY_WEIGHTS = {
    "must_have": 3.0,
    "good_to_have": 2.0,
    "bonus": 1.0,
}


def _compute_structured_score(requirement_evals: list[dict]) -> dict:
    """Deterministically compute weighted score from LLM match-level evaluations."""
    if not requirement_evals:
        return {"score": 5, "confidence": "low", "flagged_for_review": True,
                "requirement_scores": []}

    weighted_sum = 0.0
    max_possible = 0.0
    requirement_scores = []
    evidence_count = 0

    for ev in requirement_evals:
        match_level = ev.get("match_level", "no_match")
        if match_level not in _MATCH_LEVEL_SCORES:
            match_level = "no_match"

        numeric_score = _MATCH_LEVEL_SCORES[match_level]
        category = ev.get("category", "must_have")
        weight = _CATEGORY_WEIGHTS.get(category, 1.0)

        weighted_sum += numeric_score * weight
        max_possible += 10 * weight

        if match_level != "no_match":
            evidence_count += 1

        requirement_scores.append({
            "requirement": ev.get("requirement", ""),
            "category": category,
            "score": numeric_score,
            "match_level": match_level,
            "explanation": ev.get("explanation", ""),
            "evidence_snippet": ev.get("evidence_snippet"),
        })

    final_score = round((weighted_sum / max_possible) * 10, 1) if max_possible > 0 else 5

    # Confidence based on evidence coverage
    coverage = evidence_count / len(requirement_evals) if requirement_evals else 0
    if coverage >= 0.7:
        confidence = "high"
    elif coverage >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    flagged = confidence == "low" or final_score < 4

    return {
        "score": final_score,
        "confidence": confidence,
        "flagged_for_review": flagged,
        "requirement_scores": requirement_scores,
    }


def score_candidate(jd_text: str, requirements: list[dict],
                    evidence_chunks: list[dict], candidate_profile: dict = None) -> dict:
    """Score a candidate using hybrid LLM evaluation + deterministic structured scoring.

    Stage 1: LLM evaluates each requirement contextually (strong/moderate/weak/no match).
    Stage 2: Deterministic code maps match levels to scores and computes weighted total.

    Args:
        jd_text: Job description text.
        requirements: List of requirement dicts with 'text' and 'category'.
        evidence_chunks: Retrieved evidence chunks from RAG.
        candidate_profile: Structured dict with extracted fields and/or summary.
    """
    req_lines = []
    for i, req in enumerate(requirements):
        cat = req.get("category", "must_have")
        cat_label = cat.replace("_", " ").title()
        req_lines.append(f'{i+1}. [{cat_label}] {req["text"]}')
    req_str = "\n".join(req_lines) if req_lines else "No specific requirements listed. Evaluate based on overall JD match."

    evidence_str = ""
    for i, chunk in enumerate(evidence_chunks[:15]):
        match_type = chunk.get("match_type", "semantic")
        section = chunk.get("section", "unknown")
        evidence_str += f"\n--- Evidence {i+1} (section: {section}, match: {match_type}) ---\n"
        evidence_str += chunk.get("chunk_text", chunk.get("text", ""))[:500]
        if chunk.get("matched_requirement"):
            evidence_str += f"\n[Retrieved for requirement: {chunk['matched_requirement'][:100]}]"
        evidence_str += "\n"

    if not evidence_str:
        evidence_str = "(No specific evidence chunks retrieved.)"

    # Build candidate profile string from structured data
    profile_str = ""
    if candidate_profile:
        for key, value in candidate_profile.items():
            if value is not None:
                label = key.replace("_", " ").title()
                if isinstance(value, list):
                    profile_str += f"- {label}: {', '.join(str(v) for v in value)}\n"
                else:
                    profile_str += f"- {label}: {value}\n"

    if not profile_str:
        profile_str = "(No structured profile available. Rely on evidence chunks.)"

    prompt = f"""You are an expert recruiter AI evaluating a candidate against job requirements.
Use ONLY the evidence chunks and candidate profile provided.

JOB DESCRIPTION:
\"\"\"
{jd_text[:3000]}
\"\"\"

REQUIREMENTS (with priority categories):
{req_str}

RELEVANT EVIDENCE FROM RESUME:
{evidence_str}

CANDIDATE PROFILE:
{profile_str}

EVALUATION INSTRUCTIONS:
For EACH requirement, decide the match level based on overall context, NOT exact keywords.

Match levels:
- "strong_match": The candidate clearly meets this requirement. This includes equivalent technologies, directly related hands-on experience, or practical evidence through projects/work that demonstrates this competency. If the JD asks for X and the candidate has significant experience with X or a very close equivalent, this is a strong match.
- "moderate_match": The candidate has adjacent, transferable, or partially related experience. For example, the candidate worked on similar problems, used related tools, or has overlapping responsibilities that show they could fulfill this requirement with minimal ramp-up.
- "weak_match": The candidate has some tangential relevance. There is slight evidence or distant experience that touches on this requirement, but it is not a core strength.
- "no_match": There is no meaningful evidence in the resume for this requirement. Reserve this for cases where the candidate truly has nothing related.

IMPORTANT GUIDELINES:
- Be fair but slightly generous. Reward contextual relevance and transferable experience.
- Treat equivalent technologies as strong matches (e.g., PostgreSQL experience for a SQL requirement, React Native for mobile development).
- Give reasonable credit for practical project work even if phrasing differs from the JD.
- If the candidate shows they have worked on similar problems or responsibilities, give credit even if exact keywords are absent.
- Reserve no_match for truly unrelated candidates, not for wording differences.

Also write a 2-3 sentence summary justifying the overall fit.

Return ONLY valid JSON in exactly this format:
{{
  "summary": "<2-3 sentence overall assessment>",
  "requirement_evaluations": [
    {{
      "requirement": "<requirement text>",
      "category": "<must_have|good_to_have|bonus>",
      "match_level": "<strong_match|moderate_match|weak_match|no_match>",
      "explanation": "<why this match level, citing specific evidence>",
      "evidence_snippet": "<relevant text from evidence or null>"
    }}
  ]
}}"""

    response = _chat(prompt)
    llm_result = _parse_json_response(response)

    # Stage 2: Deterministic structured scoring
    requirement_evals = llm_result.get("requirement_evaluations", [])
    structured = _compute_structured_score(requirement_evals)

    summary = llm_result.get("summary", "Score generated.")
    structured["summary"] = summary

    return structured


def check_resume_quality(resume_text: str) -> dict:
    prompt = f"""Evaluate the quality of this resume text. Consider:
1. Completeness (contact info, experience, education, skills)
2. Clarity and readability
3. Professional formatting indicators
4. Specificity (metrics, achievements vs. vague descriptions)
5. Length appropriateness

Resume text:
\"\"\"
{resume_text[:5000]}
\"\"\"

Return ONLY valid JSON:
{{
  "quality_score": <0-10>,
  "issues": ["<issue 1>", "<issue 2>"],
  "suggestions": ["<suggestion 1>", "<suggestion 2>"],
  "completeness": {{
    "has_contact_info": <true|false>,
    "has_experience": <true|false>,
    "has_education": <true|false>,
    "has_skills": <true|false>,
    "has_summary": <true|false>
  }}
}}"""

    response = _chat(prompt)
    result = _parse_json_response(response)
    result.setdefault("quality_score", 5)
    result.setdefault("issues", [])
    result.setdefault("suggestions", [])
    return result


def generate_phone_screen_prep(jd_text: str, resume_text: str,
                                score_result: dict) -> dict:
    weak_areas = []
    strong_areas = []
    for req_score in score_result.get("requirement_scores", []):
        if req_score.get("score", 10) < 6:
            weak_areas.append(req_score.get("requirement", ""))
        elif req_score.get("score", 0) >= 7:
            strong_areas.append(req_score.get("requirement", ""))

    weak_str = "\n".join(f"- {w}" for w in weak_areas) if weak_areas else "None identified."
    strong_str = "\n".join(f"- {s}" for s in strong_areas) if strong_areas else "None identified."

    prompt = f"""You are helping an HR recruiter prepare for a quick phone screening call. Write in simple, non-technical language. Be concise.

JOB DESCRIPTION:
\"\"\"
{jd_text[:2000]}
\"\"\"

CANDIDATE'S RESUME:
\"\"\"
{resume_text[:2000]}
\"\"\"

SCORE: {score_result.get('score', 'N/A')}/10

STRONG AREAS:
{strong_str}

WEAK AREAS:
{weak_str}

Generate a phone screen prep sheet. Return ONLY valid JSON with:

1. "questions": 4-5 simple phone screening questions. Write each question as you would actually say it on the phone. Keep rationale to one short sentence. Use simple language.

2. "call_notes": A short paragraph (3-5 sentences) with practical cues for the recruiter. Write it like advice from a senior colleague. Include things like:
   - What to listen for if the candidate describes their experience (does it match the resume?)
   - What gaps or missing experience to watch out for
   - Any claims on the resume that seem vague or hard to verify
   - Red flags to notice during the call (fumbling on basic details, not being able to explain their actual work, experience timeline that doesn't add up)
   Keep it conversational and practical.

Return ONLY valid JSON:
{{
  "questions": [
    {{
      "question": "<a natural phone screening question>",
      "rationale": "<one short sentence on why>"
    }}
  ],
  "call_notes": "<a short paragraph with practical cues>"
}}"""

    response = _chat(prompt)
    result = _parse_json_response(response)
    if not isinstance(result, dict):
        return {"questions": [], "call_notes": ""}
    result.setdefault("questions", [])
    result.setdefault("call_notes", "")
    return result


def generate_candidate_summary(resume_text: str) -> str:
    prompt = f"""Write a concise 3-4 sentence professional summary of this candidate
based on their resume. Focus on: years of experience, primary expertise,
notable achievements, and overall profile strength.

Resume text:
\"\"\"
{resume_text[:4000]}
\"\"\"

Return ONLY the summary text, no JSON."""

    return _chat(prompt)
