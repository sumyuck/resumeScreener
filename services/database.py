"""
Database service: Supabase client and CRUD operations.
"""

import os
import psycopg2
import psycopg2.extras
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()


def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


def get_db_connection():
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise EnvironmentError("SUPABASE_DB_URL must be set in .env")
    return psycopg2.connect(db_url)


# --- Users ---

def get_or_create_default_user(client: Client) -> dict:
    resp = client.table("users").select("*").eq("email", "recruiter@sprinto.com").execute()
    if resp.data:
        return resp.data[0]
    resp = client.table("users").insert({
        "name": "Default Recruiter",
        "email": "recruiter@sprinto.com"
    }).execute()
    return resp.data[0]


# --- Roles ---

def list_roles(client: Client, status: str = None) -> list[dict]:
    q = client.table("roles").select("*").order("created_at", desc=True)
    if status:
        q = q.eq("status", status)
    return q.execute().data


def get_role(client: Client, role_id: str) -> dict | None:
    resp = client.table("roles").select("*").eq("id", role_id).execute()
    return resp.data[0] if resp.data else None


def create_role(client: Client, title: str, department: str, jd_text: str,
                requirements: list, created_by: str = None) -> dict:
    payload = {
        "title": title,
        "department": department,
        "jd_text": jd_text,
        "requirements": requirements,
    }
    if created_by:
        payload["created_by"] = created_by
    return client.table("roles").insert(payload).execute().data[0]


def update_role(client: Client, role_id: str, updates: dict) -> dict:
    return client.table("roles").update(updates).eq("id", role_id).execute().data[0]


def delete_role(client: Client, role_id: str):
    client.table("roles").delete().eq("id", role_id).execute()


# --- Resumes ---

def list_resumes(client: Client) -> list[dict]:
    return client.table("resumes").select("*").order("created_at", desc=True).execute().data


def get_resume(client: Client, resume_id: str) -> dict | None:
    resp = client.table("resumes").select("*").eq("id", resume_id).execute()
    return resp.data[0] if resp.data else None


def find_resume_by_hash(client: Client, file_hash: str) -> list[dict]:
    return client.table("resumes").select("*").eq("file_hash", file_hash).execute().data


def find_resume_by_text_hash(client: Client, text_hash: str) -> list[dict]:
    """Find resumes with matching normalized text hash (format-independent dedup)."""
    return client.table("resumes").select("*").eq("text_hash", text_hash).execute().data


def insert_resume(client: Client, data: dict) -> dict:
    return client.table("resumes").insert(data).execute().data[0]


def update_resume(client: Client, resume_id: str, updates: dict) -> dict:
    return client.table("resumes").update(updates).eq("id", resume_id).execute().data[0]


def delete_resume(client: Client, conn, resume_id: str):
    """Delete a resume and all associated data (chunks, embeddings, scan results, feedback, etc.)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunk_embeddings WHERE resume_id = %s", (resume_id,))
    conn.commit()
    client.table("resume_chunks").delete().eq("resume_id", resume_id).execute()
    client.table("extracted_fields").delete().eq("resume_id", resume_id).execute()
    client.table("scan_results").delete().eq("resume_id", resume_id).execute()
    client.table("recruiter_feedback").delete().eq("resume_id", resume_id).execute()
    client.table("duplicate_flags").delete().eq("resume_id", resume_id).execute()
    client.table("resumes").delete().eq("id", resume_id).execute()


# --- Extraction Configs ---

def get_default_config(client: Client) -> dict | None:
    resp = client.table("extraction_configs").select("*").eq("is_default", True).execute()
    return resp.data[0] if resp.data else None


def list_configs(client: Client) -> list[dict]:
    return client.table("extraction_configs").select("*").order("created_at", desc=True).execute().data


def upsert_config(client: Client, name: str, fields: list, is_default: bool = False) -> dict:
    if is_default:
        # Unset is_default on all existing configs first
        existing = client.table("extraction_configs").select("id").eq("is_default", True).execute().data
        for cfg in existing:
            client.table("extraction_configs").update({"is_default": False}).eq("id", cfg["id"]).execute()
    return client.table("extraction_configs").insert({
        "name": name,
        "fields": fields,
        "is_default": is_default,
    }).execute().data[0]


def set_default_config(client: Client, config_id: str):
    """Set a config as the default, unsetting all others."""
    existing = client.table("extraction_configs").select("id").eq("is_default", True).execute().data
    for cfg in existing:
        client.table("extraction_configs").update({"is_default": False}).eq("id", cfg["id"]).execute()
    client.table("extraction_configs").update({"is_default": True}).eq("id", config_id).execute()


def update_config(client: Client, config_id: str, name: str, fields: list) -> dict:
    return client.table("extraction_configs").update({
        "name": name,
        "fields": fields,
    }).eq("id", config_id).execute().data[0]


# --- Extracted Fields ---

def get_extracted_fields(client: Client, resume_id: str) -> dict | None:
    resp = (client.table("extracted_fields")
            .select("*")
            .eq("resume_id", resume_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute())
    return resp.data[0] if resp.data else None


def save_extracted_fields(client: Client, resume_id: str, config_id: str, fields: dict) -> dict:
    return client.table("extracted_fields").insert({
        "resume_id": resume_id,
        "config_id": config_id,
        "fields": fields,
    }).execute().data[0]


# --- Chunks and Embeddings ---

def save_chunks(client: Client, resume_id: str, chunks: list[dict]) -> list[dict]:
    rows = [{"resume_id": resume_id, **c} for c in chunks]
    return client.table("resume_chunks").insert(rows).execute().data


def get_chunks(client: Client, resume_id: str) -> list[dict]:
    return (client.table("resume_chunks")
            .select("*")
            .eq("resume_id", resume_id)
            .order("chunk_index")
            .execute().data)


def save_embedding(conn, chunk_id: str, resume_id: str, embedding: list[float]):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chunk_embeddings (chunk_id, resume_id, embedding) VALUES (%s, %s, %s::vector)",
            (chunk_id, resume_id, str(embedding))
        )
    conn.commit()


def save_embeddings_batch(conn, rows: list[tuple]):
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO chunk_embeddings (chunk_id, resume_id, embedding) VALUES %s",
            [(r[0], r[1], str(r[2])) for r in rows],
            template="(%s, %s, %s::vector)"
        )
    conn.commit()


def similarity_search(conn, query_embedding: list[float], top_k: int = 10,
                      resume_ids: list[str] = None) -> list[dict]:
    query = """
        SELECT ce.chunk_id, ce.resume_id, rc.chunk_text, rc.section,
               1 - (ce.embedding <=> %s::vector) AS similarity
        FROM chunk_embeddings ce
        JOIN resume_chunks rc ON rc.id = ce.chunk_id
    """
    params: list = [str(query_embedding)]
    if resume_ids:
        query += " WHERE ce.resume_id = ANY(%s)"
        params.append(resume_ids)
    query += " ORDER BY ce.embedding <=> %s::vector LIMIT %s"
    params.extend([str(query_embedding), top_k])

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, params)
        return cur.fetchall()


def delete_chunks_and_embeddings(client: Client, conn, resume_id: str):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunk_embeddings WHERE resume_id = %s", (resume_id,))
    conn.commit()
    client.table("resume_chunks").delete().eq("resume_id", resume_id).execute()


# --- Scan Results ---

def get_scan_results_for_role(client: Client, role_id: str) -> list[dict]:
    return (client.table("scan_results")
            .select("*, resumes(id, filename, candidate_name, candidate_email)")
            .eq("role_id", role_id)
            .order("score", desc=True)
            .execute().data)


def get_scan_result(client: Client, resume_id: str, role_id: str) -> dict | None:
    resp = (client.table("scan_results")
            .select("*")
            .eq("resume_id", resume_id)
            .eq("role_id", role_id)
            .execute())
    return resp.data[0] if resp.data else None


def get_scan_results_for_resume(client: Client, resume_id: str) -> list[dict]:
    return (client.table("scan_results")
            .select("*, roles(id, title, department)")
            .eq("resume_id", resume_id)
            .order("score", desc=True)
            .execute().data)


def upsert_scan_result(client: Client, data: dict) -> dict:
    return client.table("scan_results").upsert(
        data, on_conflict="resume_id,role_id"
    ).execute().data[0]


# --- Duplicate Flags ---

def get_duplicates_for_resume(client: Client, resume_id: str) -> list[dict]:
    return (client.table("duplicate_flags")
            .select("*")
            .eq("resume_id", resume_id)
            .execute().data)


def flag_duplicate(client: Client, resume_id: str, duplicate_of: str,
                   role_id: str, flag_type: str, similarity: float = None) -> dict:
    return client.table("duplicate_flags").insert({
        "resume_id": resume_id,
        "duplicate_of": duplicate_of,
        "role_id": role_id,
        "flag_type": flag_type,
        "similarity": similarity,
    }).execute().data[0]


# --- Recruiter Feedback ---

def get_feedback(client: Client, resume_id: str, role_id: str) -> dict | None:
    resp = (client.table("recruiter_feedback")
            .select("*")
            .eq("resume_id", resume_id)
            .eq("role_id", role_id)
            .execute())
    return resp.data[0] if resp.data else None


def save_feedback(client: Client, resume_id: str, role_id: str,
                  decision: str, notes: str = "", decided_by: str = None) -> dict:
    return client.table("recruiter_feedback").upsert({
        "resume_id": resume_id,
        "role_id": role_id,
        "decision": decision,
        "notes": notes,
        "decided_by": decided_by,
    }, on_conflict="resume_id,role_id").execute().data[0]


# --- Scan History ---

def create_scan_history(client: Client, role_id: str, resume_count: int,
                        config_id: str = None, scan_type: str = "manual",
                        triggered_by: str = None) -> dict:
    return client.table("scan_history").insert({
        "role_id": role_id,
        "resume_count": resume_count,
        "config_id": config_id,
        "scan_type": scan_type,
        "triggered_by": triggered_by,
    }).execute().data[0]


def complete_scan_history(client: Client, scan_id: str, status: str = "completed",
                          notes: str = None):
    updates = {"status": status, "completed_at": "now()"}
    if notes:
        updates["notes"] = notes
    client.table("scan_history").update(updates).eq("id", scan_id).execute()


def list_scan_history(client: Client, role_id: str = None) -> list[dict]:
    q = client.table("scan_history").select("*, roles(title)").order("started_at", desc=True)
    if role_id:
        q = q.eq("role_id", role_id)
    return q.execute().data
