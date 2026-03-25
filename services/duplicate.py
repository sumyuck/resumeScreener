"""
Duplicate detection: exact hash matching and fuzzy embedding similarity.
"""

from services.database import (
    find_resume_by_hash, flag_duplicate, get_chunks,
    similarity_search as db_similarity_search,
)
from services.embeddings import generate_embedding


def check_duplicates(client, conn, resume_id: str, file_hash: str,
                     resume_text: str, role_id: str = None) -> list[dict]:
    flags = []

    exact_matches = find_resume_by_hash(client, file_hash)
    for match in exact_matches:
        if match["id"] != resume_id:
            try:
                flag = flag_duplicate(
                    client,
                    resume_id=resume_id,
                    duplicate_of=match["id"],
                    role_id=role_id,
                    flag_type="exact",
                    similarity=1.0,
                )
                flags.append(flag)
            except Exception:
                flags.append({
                    "resume_id": resume_id,
                    "duplicate_of": match["id"],
                    "flag_type": "exact",
                    "similarity": 1.0,
                })

    try:
        fuzzy_flags = _check_fuzzy_duplicates(client, conn, resume_id, resume_text, role_id)
        flags.extend(fuzzy_flags)
    except Exception:
        pass

    return flags


def _check_fuzzy_duplicates(client, conn, resume_id: str,
                             resume_text: str, role_id: str = None,
                             threshold: float = 0.92) -> list[dict]:
    flags = []
    sample_text = resume_text[:1500]

    try:
        query_embedding = generate_embedding(sample_text)
    except Exception:
        return []

    results = db_similarity_search(conn, query_embedding, top_k=10)

    for result in results:
        sim = float(result.get("similarity", 0))
        other_resume_id = result.get("resume_id")

        if other_resume_id == resume_id:
            continue

        if sim >= threshold:
            try:
                flag = flag_duplicate(
                    client,
                    resume_id=resume_id,
                    duplicate_of=other_resume_id,
                    role_id=role_id,
                    flag_type="possible",
                    similarity=sim,
                )
                flags.append(flag)
            except Exception:
                flags.append({
                    "resume_id": resume_id,
                    "duplicate_of": other_resume_id,
                    "flag_type": "possible",
                    "similarity": sim,
                })

    return flags
