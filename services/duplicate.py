"""
Duplicate detection: exact hash matching (file + text), and fuzzy embedding similarity.
"""

import hashlib
import re

from services.database import (
    find_resume_by_hash, find_resume_by_text_hash, flag_duplicate, get_chunks,
    similarity_search as db_similarity_search,
)
from services.embeddings import generate_embedding


def compute_text_hash(raw_text: str) -> str:
    """Compute a format-independent hash of resume text content.

    Normalizes text (lowercase, collapse whitespace, strip punctuation)
    before hashing so that the same resume in PDF vs DOCX produces the same hash.
    """
    normalized = raw_text.lower()
    normalized = re.sub(r'[^\w\s]', '', normalized)  # strip punctuation
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def check_exact_duplicate_before_upload(client, file_hash: str, raw_text: str) -> dict | None:
    """Pre-upload check: returns the existing resume if an exact duplicate is found.

    Checks both file hash (same file re-uploaded) and text hash (same content,
    different format like PDF vs DOCX).

    Returns:
        The existing resume dict if duplicate found, None otherwise.
    """
    # Check file hash first (exact same file)
    file_matches = find_resume_by_hash(client, file_hash)
    if file_matches:
        return file_matches[0]

    # Check text hash (same content, different format)
    text_hash = compute_text_hash(raw_text)
    text_matches = find_resume_by_text_hash(client, text_hash)
    if text_matches:
        return text_matches[0]

    return None


def check_duplicates(client, conn, resume_id: str, file_hash: str,
                     resume_text: str, role_id: str = None) -> list[dict]:
    """Post-upload duplicate check: flag exact and fuzzy duplicates."""
    flags = []

    # Exact file hash matches
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

    # Exact text hash matches (cross-format)
    text_hash = compute_text_hash(resume_text)
    text_matches = find_resume_by_text_hash(client, text_hash)
    seen_ids = {f.get("duplicate_of") for f in flags}
    for match in text_matches:
        if match["id"] != resume_id and match["id"] not in seen_ids:
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

    # Fuzzy embedding similarity (only if no exact matches found)
    if not flags:
        try:
            fuzzy_flags = _check_fuzzy_duplicates(client, conn, resume_id, resume_text, role_id)
            flags.extend(fuzzy_flags)
        except Exception:
            pass

    return flags


def _check_fuzzy_duplicates(client, conn, resume_id: str,
                             resume_text: str, role_id: str = None,
                             threshold: float = 0.95) -> list[dict]:
    """Fuzzy duplicate check using embedding similarity.

    Uses a higher threshold (0.95) than before to avoid false positives
    from candidates with similar names but different experience.
    """
    flags = []
    # Use a substantial text sample for better semantic comparison
    sample_text = resume_text[:2000]

    try:
        query_embedding = generate_embedding(sample_text)
    except Exception:
        return []

    results = db_similarity_search(conn, query_embedding, top_k=5)

    # Deduplicate by resume_id (we may get multiple chunks from same resume)
    seen_resume_ids = set()
    for result in results:
        sim = float(result.get("similarity", 0))
        other_resume_id = result.get("resume_id")

        if other_resume_id == resume_id or other_resume_id in seen_resume_ids:
            continue
        seen_resume_ids.add(other_resume_id)

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
