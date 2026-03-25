"""
Upload Resumes: multi-file upload with parsing, embedding, and duplicate detection.
"""

import streamlit as st
import time

st.set_page_config(page_title="Upload Resumes", page_icon="R", layout="wide")

st.markdown("# Upload Resumes")

try:
    from services.database import (
        get_supabase_client, get_db_connection, insert_resume, update_resume,
        save_chunks, get_default_config, save_extracted_fields,
        get_or_create_default_user, list_resumes, delete_resume,
    )
    from services.database import save_embeddings_batch
    from services.parser import parse_resume, chunk_text
    from services.embeddings import generate_embeddings_batch
    from services.ai_engine import extract_fields, check_resume_quality
    from services.duplicate import check_duplicates
    from services.utils import compute_file_hash, load_default_extraction_config

    client = get_supabase_client()
    user = get_or_create_default_user(client)

    # File uploader
    uploaded_files = st.file_uploader(
        "Drop PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    with st.expander("Upload options", expanded=False):
        run_quality_check = st.checkbox("Run quality check", value=True)
        run_extraction = st.checkbox("Extract fields", value=True)
        run_duplicate_check = st.checkbox("Check for duplicates", value=True)

    # Process uploads
    if uploaded_files and st.button("Process Uploads", type="primary"):
        progress = st.progress(0, text="Starting...")
        results = []
        total = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files):
            file_name = uploaded_file.name
            step_prefix = f"[{idx+1}/{total}] {file_name}"
            progress.progress((idx) / total, text=f"{step_prefix}: Reading...")

            try:
                file_bytes = uploaded_file.read()
                file_hash = compute_file_hash(file_bytes)
                file_type = file_name.rsplit('.', 1)[-1].lower()

                progress.progress((idx + 0.15) / total, text=f"{step_prefix}: Parsing...")
                raw_text = parse_resume(file_bytes, file_name)

                progress.progress((idx + 0.25) / total, text=f"{step_prefix}: Saving...")
                resume_record = insert_resume(client, {
                    "filename": file_name,
                    "file_type": file_type,
                    "file_hash": file_hash,
                    "raw_text": raw_text,
                    "status": "parsing",
                    "uploaded_by": user["id"],
                })
                resume_id = resume_record["id"]

                progress.progress((idx + 0.35) / total, text=f"{step_prefix}: Chunking...")
                chunks = chunk_text(raw_text)
                saved_chunks = save_chunks(client, resume_id, chunks)

                progress.progress((idx + 0.45) / total, text=f"{step_prefix}: Embedding...")
                chunk_texts = [c["chunk_text"] for c in chunks]
                embeddings = generate_embeddings_batch(chunk_texts)

                conn = get_db_connection()
                try:
                    embedding_rows = [
                        (saved_chunks[i]["id"], resume_id, embeddings[i])
                        for i in range(len(saved_chunks))
                    ]
                    save_embeddings_batch(conn, embedding_rows)
                finally:
                    conn.close()

                extracted = {}
                if run_extraction:
                    progress.progress((idx + 0.6) / total, text=f"{step_prefix}: Extracting fields...")
                    config = get_default_config(client)
                    if config:
                        config_fields = config.get("fields", [])
                        config_id = config["id"]
                    else:
                        default = load_default_extraction_config()
                        config_fields = default.get("fields", [])
                        config_id = None

                    extracted = extract_fields(raw_text, config_fields)
                    if config_id:
                        save_extracted_fields(client, resume_id, config_id, extracted)

                    update_data = {"status": "parsed"}
                    if extracted.get("candidate_name"):
                        update_data["candidate_name"] = extracted["candidate_name"]
                    if extracted.get("email"):
                        update_data["candidate_email"] = extracted["email"]
                    update_resume(client, resume_id, update_data)

                quality = {}
                if run_quality_check:
                    progress.progress((idx + 0.75) / total, text=f"{step_prefix}: Quality check...")
                    quality = check_resume_quality(raw_text)
                    update_resume(client, resume_id, {
                        "quality_score": quality.get("quality_score"),
                        "quality_notes": str(quality.get("issues", [])),
                    })

                duplicates = []
                if run_duplicate_check:
                    progress.progress((idx + 0.85) / total, text=f"{step_prefix}: Checking duplicates...")
                    conn = get_db_connection()
                    try:
                        duplicates = check_duplicates(client, conn, resume_id, file_hash, raw_text)
                    finally:
                        conn.close()

                update_resume(client, resume_id, {"status": "parsed"})

                results.append({
                    "file": file_name,
                    "status": "Success",
                    "name": extracted.get("candidate_name", "N/A"),
                    "chunks": len(chunks),
                    "quality": quality.get("quality_score", "N/A"),
                    "duplicates": len(duplicates),
                })

            except Exception as e:
                # Set status to error if resume was created
                try:
                    if 'resume_id' in locals():
                        update_resume(client, resume_id, {"status": "error"})
                except Exception:
                    pass
                results.append({
                    "file": file_name,
                    "status": f"Error: {str(e)[:80]}",
                    "name": "N/A",
                    "chunks": 0,
                    "quality": "N/A",
                    "duplicates": 0,
                })

        progress.progress(1.0, text="Processing complete")

        st.markdown("### Upload Results")
        import pandas as pd
        df = pd.DataFrame(results)
        df.columns = ["File", "Status", "Candidate", "Chunks", "Quality", "Duplicates"]
        st.dataframe(df, use_container_width=True, hide_index=True)

        success_count = sum(1 for r in results if r["status"] == "Success")
        if success_count == total:
            st.success(f"All {total} resumes processed successfully.")
        else:
            st.warning(f"Processed {success_count}/{total} resumes. Check errors above.")

    # Existing resumes
    st.markdown("---")
    st.markdown("### Uploaded Resumes")

    resumes = list_resumes(client)

    if resumes:
        # Delete confirmation state
        if "delete_resume_id" not in st.session_state:
            st.session_state.delete_resume_id = None

        import pandas as pd
        for r in resumes:
            col1, col2, col3, col4, col5 = st.columns([3, 2, 1, 1, 1])
            with col1:
                st.markdown(f"**{r.get('candidate_name') or 'Unknown'}**")
                st.caption(r.get("filename", ""))
            with col2:
                st.caption(f"{r.get('file_type', '').upper()} | {r.get('created_at', '')[:10]}")
            with col3:
                status = r.get("status", "")
                st.caption(status.title())
            with col4:
                q = r.get("quality_score")
                st.caption(f"{q:.1f}" if q else "N/A")
            with col5:
                if st.session_state.delete_resume_id == r["id"]:
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Yes", key=f"confirm_del_{r['id']}", type="primary"):
                            conn = get_db_connection()
                            try:
                                delete_resume(client, conn, r["id"])
                            finally:
                                conn.close()
                            st.session_state.delete_resume_id = None
                            st.rerun()
                    with c2:
                        if st.button("No", key=f"cancel_del_{r['id']}"):
                            st.session_state.delete_resume_id = None
                            st.rerun()
                else:
                    if st.button("Delete", key=f"del_{r['id']}"):
                        st.session_state.delete_resume_id = r["id"]
                        st.rerun()

        st.caption(f"{len(resumes)} resumes total")
    else:
        st.caption("No resumes uploaded yet.")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.caption("Check your .env configuration.")
