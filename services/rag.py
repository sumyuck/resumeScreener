"""
RAG service: hybrid retrieval combining pgvector semantic search with keyword matching.
Uses requirement-centric retrieval and filters low-signal chunks.
"""

import re
from services.embeddings import generate_query_embedding
from services.database import similarity_search, get_chunks
from services.utils import extract_skills_from_text

# Minimum chunk length to be considered useful (filters header-only chunks)
_MIN_CHUNK_CHARS = 50


def _is_low_signal(chunk: dict) -> bool:
    """Return True if chunk is header-only or too short to be useful."""
    text = chunk.get("chunk_text", "").strip()
    # Strip section label prefix like "[Skills] "
    cleaned = re.sub(r'^\[[\w\s]+\]\s*', '', text).strip()
    return len(cleaned) < _MIN_CHUNK_CHARS


def hybrid_retrieve(conn, client, jd_text: str, resume_id: str,
                    requirements: list[dict] = None, top_k: int = 10) -> list[dict]:
    """Requirement-centric hybrid retrieval: fetch evidence per requirement, then merge."""
    all_chunks = get_chunks(client, resume_id)
    # Filter out low-signal chunks upfront
    useful_chunks = [c for c in all_chunks if not _is_low_signal(c)]

    jd_skills = extract_skills_from_text(jd_text)
    if requirements:
        for req in requirements:
            req_skills = extract_skills_from_text(req.get("text", ""))
            jd_skills.extend(req_skills)
    jd_skills = list(set(jd_skills))

    # Requirement-centric semantic retrieval
    semantic_results = []
    if requirements:
        semantic_results = _requirement_centric_search(
            conn, requirements, resume_id, top_k_per_req=5
        )
    else:
        # Fallback: single JD-level search
        semantic_results = _semantic_search(conn, jd_text, resume_id, top_k=top_k * 2)

    # Keyword matching on filtered chunks
    keyword_results = _keyword_match(useful_chunks, jd_skills, jd_text)

    # Merge and rank
    merged = _merge_and_rank(semantic_results, keyword_results, top_k)

    # Final filter: remove any low-signal chunks that slipped through
    merged = [r for r in merged if not _is_low_signal(r)]

    return merged


def _requirement_centric_search(conn, requirements: list[dict], resume_id: str,
                                 top_k_per_req: int = 5) -> list[dict]:
    """Fetch evidence separately for each requirement and deduplicate."""
    all_results = []
    seen_chunks = set()

    for req in requirements:
        req_text = req.get("text", "")
        if not req_text:
            continue

        try:
            query_emb = generate_query_embedding(req_text)
            results = similarity_search(
                conn, query_emb, top_k=top_k_per_req, resume_ids=[resume_id]
            )
            for r in results:
                chunk_key = r.get("chunk_text", "")[:100]
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    r["match_type"] = "semantic"
                    r["semantic_score"] = float(r.get("similarity", 0))
                    r["matched_requirement"] = req_text
                    r["requirement_category"] = req.get("category", "must_have")
                    all_results.append(r)
        except Exception:
            continue

    return all_results


def _semantic_search(conn, query_text: str, resume_id: str,
                     top_k: int = 20) -> list[dict]:
    try:
        query_embedding = generate_query_embedding(query_text)
        results = similarity_search(
            conn, query_embedding, top_k=top_k, resume_ids=[resume_id]
        )
        for r in results:
            r["match_type"] = "semantic"
            r["semantic_score"] = float(r.get("similarity", 0))
        return results
    except Exception:
        return []


def _keyword_match(chunks: list[dict], jd_skills: list[str],
                   jd_text: str) -> list[dict]:
    if not jd_skills and not jd_text:
        return []

    jd_terms = set(re.findall(r'\b[a-z][a-z0-9.#+]+\b', jd_text.lower()))
    results = []

    # Section relevance boost: experience/projects/skills are more valuable
    _section_boost = {
        "experience": 1.2, "projects": 1.15, "skills": 1.1,
        "summary": 1.05, "education": 1.0, "achievements": 1.05,
    }

    for chunk in chunks:
        text = chunk.get("chunk_text", "").lower()
        if not text:
            continue

        matched_skills = [skill for skill in jd_skills if skill.lower() in text]
        skill_score = len(matched_skills) / max(len(jd_skills), 1)

        chunk_terms = set(re.findall(r'\b[a-z][a-z0-9.#+]+\b', text))
        common_terms = jd_terms & chunk_terms
        term_score = len(common_terms) / max(len(jd_terms), 1)

        keyword_score = 0.7 * skill_score + 0.3 * term_score

        # Apply section relevance boost
        section = chunk.get("section", "general")
        boost = _section_boost.get(section, 1.0)
        keyword_score *= boost

        if keyword_score > 0.05:
            results.append({
                "chunk_id": chunk.get("id"),
                "resume_id": chunk.get("resume_id"),
                "chunk_text": chunk.get("chunk_text"),
                "section": chunk.get("section"),
                "match_type": "keyword",
                "keyword_score": keyword_score,
                "matched_skills": matched_skills,
            })

    results.sort(key=lambda x: x["keyword_score"], reverse=True)
    return results


def _merge_and_rank(semantic_results: list[dict], keyword_results: list[dict],
                    top_k: int = 10) -> list[dict]:
    k = 60
    score_map: dict[str, dict] = {}

    for rank, result in enumerate(semantic_results):
        key = result.get("chunk_text", "")[:100]
        if key not in score_map:
            score_map[key] = {**result, "rrf_score": 0, "match_type": "semantic"}
        score_map[key]["rrf_score"] += 1.0 / (k + rank + 1)
        score_map[key]["semantic_score"] = result.get("semantic_score", 0)

    for rank, result in enumerate(keyword_results):
        key = result.get("chunk_text", "")[:100]
        if key not in score_map:
            score_map[key] = {**result, "rrf_score": 0}
        score_map[key]["rrf_score"] += 1.0 / (k + rank + 1)
        score_map[key]["keyword_score"] = result.get("keyword_score", 0)
        score_map[key]["matched_skills"] = result.get("matched_skills", [])
        if score_map[key].get("match_type") == "semantic":
            score_map[key]["match_type"] = "both"
        else:
            score_map[key]["match_type"] = result.get("match_type", "keyword")

    # Section diversity bonus: slightly boost underrepresented sections
    section_counts: dict[str, int] = {}
    for v in score_map.values():
        s = v.get("section", "general")
        section_counts[s] = section_counts.get(s, 0) + 1

    for v in score_map.values():
        s = v.get("section", "general")
        if section_counts.get(s, 0) <= 2:
            v["rrf_score"] *= 1.1  # Small boost for rare sections

    merged = sorted(score_map.values(), key=lambda x: x["rrf_score"], reverse=True)
    return merged[:top_k]


def retrieve_for_requirements(conn, client, requirements: list[dict],
                              resume_id: str, top_k_per_req: int = 3) -> list[dict]:
    all_evidence = []
    seen_chunks = set()

    for req in requirements:
        req_text = req.get("text", "")
        if not req_text:
            continue

        try:
            query_emb = generate_query_embedding(req_text)
            results = similarity_search(
                conn, query_emb, top_k=top_k_per_req, resume_ids=[resume_id]
            )
            for r in results:
                chunk_key = r.get("chunk_text", "")[:100]
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    r["matched_requirement"] = req_text
                    r["requirement_category"] = req.get("category", "must_have")
                    r["match_type"] = "semantic"
                    all_evidence.append(r)
        except Exception:
            continue

    # Filter low-signal chunks
    all_evidence = [e for e in all_evidence if not _is_low_signal(e)]
    return all_evidence
