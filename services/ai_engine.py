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


def score_candidate(jd_text: str, requirements: list[dict],
                    evidence_chunks: list[dict], candidate_profile: dict = None) -> dict:
    """Score a candidate using structured requirements, evidence chunks, and candidate profile.

    Args:
        jd_text: Job description text.
        requirements: List of requirement dicts with 'text' and 'category'.
        evidence_chunks: Retrieved evidence chunks from RAG.
        candidate_profile: Structured dict with extracted fields and/or summary.
                          Keys may include: name, skills, experience, education, summary, etc.
    """
    req_lines = []
    for i, req in enumerate(requirements):
        cat = req.get("category", "must_have")
        cat_label = cat.replace("_", " ").title()
        req_lines.append(f'{i+1}. [{cat_label}] {req["text"]}')
    req_str = "\n".join(req_lines) if req_lines else "No specific requirements listed. Score based on overall JD match."

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

    prompt = f"""You are an expert recruiter AI assistant. Score this candidate against the job description.
Use ONLY the evidence chunks and candidate profile provided. Do NOT assume information not present in the evidence.

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

SCORING INSTRUCTIONS:
1. Score each requirement on a 0-10 scale based ONLY on the evidence provided.
2. Weight must_have requirements at 3x, good_to_have at 2x, and bonus at 1x.
3. Calculate a weighted overall score (1-10).
4. For each requirement, cite the specific evidence chunk that supports your rating.
5. Set confidence to "high" if evidence clearly covers most requirements, "low" if evidence is sparse, "medium" otherwise.
6. Set flagged_for_review to true if you are uncertain or detect issues.

Return ONLY valid JSON in exactly this format:
{{
  "score": <number 1-10>,
  "summary": "<2-3 sentence justification of the overall score>",
  "confidence": "<high|medium|low>",
  "flagged_for_review": <true|false>,
  "requirement_scores": [
    {{
      "requirement": "<requirement text>",
      "category": "<must_have|good_to_have|bonus>",
      "score": <0-10>,
      "explanation": "<why this score, citing evidence>",
      "evidence_snippet": "<relevant text from evidence or null>"
    }}
  ]
}}"""

    response = _chat(prompt)
    result = _parse_json_response(response)

    result.setdefault("score", 5)
    result.setdefault("summary", "Score generated.")
    result.setdefault("confidence", "medium")
    result.setdefault("flagged_for_review", False)
    result.setdefault("requirement_scores", [])

    return result


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
