"""
RAG service: hybrid retrieval combining pgvector semantic search with keyword matching.
"""

import re
from services.embeddings import generate_query_embedding
from services.database import similarity_search, get_chunks
from services.utils import extract_skills_from_text


def hybrid_retrieve(conn, client, jd_text: str, resume_id: str,
                    requirements: list[dict] = None, top_k: int = 10) -> list[dict]:
    semantic_results = _semantic_search(conn, jd_text, resume_id, top_k=top_k * 2)

    all_chunks = get_chunks(client, resume_id)
    jd_skills = extract_skills_from_text(jd_text)

    if requirements:
        for req in requirements:
            req_skills = extract_skills_from_text(req.get("text", ""))
            jd_skills.extend(req_skills)
    jd_skills = list(set(jd_skills))

    keyword_results = _keyword_match(all_chunks, jd_skills, jd_text)
    merged = _merge_and_rank(semantic_results, keyword_results, top_k)

    return merged


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

    return all_evidence
