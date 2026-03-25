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
    field_descriptions = []
    for field in extraction_config:
        req = "required" if field.get("required") else "optional"
        field_descriptions.append(
            f'- "{field["key"]}" ({field["label"]}, type: {field["type"]}, {req})'
        )
    fields_str = "\n".join(field_descriptions)

    prompt = f"""You are a precise resume information extractor.

Extract the following fields from the resume text below. Return ONLY valid JSON.

Fields to extract:
{fields_str}

Rules:
1. For "list" type fields, return a JSON array of strings.
2. For "number" type fields, return a numeric value or null.
3. For "text" type fields, return a string or null.
4. If a field cannot be found, return null for that field.
5. For "years_of_experience", calculate from work history if not explicitly stated.
6. Be precise. Do not invent or hallucinate information.

Resume text:
\"\"\"
{resume_text[:6000]}
\"\"\"

Return ONLY a JSON object with the field keys as keys."""

    response = _chat(prompt)
    return _parse_json_response(response)


def score_candidate(jd_text: str, requirements: list[dict],
                    evidence_chunks: list[dict], full_resume_text: str) -> dict:
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
        evidence_str += "\n"

    if not evidence_str:
        evidence_str = "(No specific evidence chunks retrieved. Use the full resume below.)"

    prompt = f"""You are an expert recruiter AI assistant. Score this candidate against the job description.

JOB DESCRIPTION:
\"\"\"
{jd_text[:3000]}
\"\"\"

REQUIREMENTS (with priority categories):
{req_str}

RELEVANT EVIDENCE FROM RESUME:
{evidence_str}

FULL RESUME (for additional context):
\"\"\"
{full_resume_text[:3000]}
\"\"\"

SCORING INSTRUCTIONS:
1. Score each requirement on a 0-10 scale.
2. Weight must_have requirements at 3x, good_to_have at 2x, and bonus at 1x.
3. Calculate a weighted overall score (1-10).
4. For each requirement, cite the specific evidence chunk that supports your rating.
5. Set confidence to "high" if the resume clearly covers most requirements, "low" if the resume is too vague or thin, "medium" otherwise.
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
      "explanation": "<why this score>",
      "evidence_snippet": "<relevant text from resume or null>"
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


def generate_interview_questions(jd_text: str, resume_text: str,
                                  score_result: dict) -> list[dict]:
    weak_areas = []
    for req_score in score_result.get("requirement_scores", []):
        if req_score.get("score", 10) < 6:
            weak_areas.append(req_score.get("requirement", ""))

    weak_str = "\n".join(f"- {w}" for w in weak_areas) if weak_areas else "No specific weak areas identified."

    prompt = f"""You are an expert recruiter preparing for a candidate interview.

JOB DESCRIPTION:
\"\"\"
{jd_text[:2000]}
\"\"\"

CANDIDATE'S RESUME:
\"\"\"
{resume_text[:2000]}
\"\"\"

CANDIDATE'S OVERALL SCORE: {score_result.get('score', 'N/A')}/10

AREAS NEEDING DEEPER ASSESSMENT:
{weak_str}

Generate 5-7 interview questions that:
1. Probe the areas where the candidate scored low or evidence was weak.
2. Validate claimed skills and experience.
3. Include a mix of technical and behavioral questions.
4. Be specific to this candidate and role, not generic.

Return ONLY valid JSON as a list:
[
  {{
    "question": "<the interview question>",
    "rationale": "<why this question is important>",
    "category": "<technical|behavioral|experience_validation|gap_probe>"
  }}
]"""

    response = _chat(prompt)
    result = _parse_json_response(response)
    if isinstance(result, list):
        return result
    return result.get("questions", []) if isinstance(result, dict) else []


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
