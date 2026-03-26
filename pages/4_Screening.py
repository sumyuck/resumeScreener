"""
Screening: select a job, scan resumes, review candidates in sorted order.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Screening", page_icon="R", layout="wide")

st.markdown("# Screening")


def _do_scan(client, role, user, rescan=False):
    from services.database import list_resumes, get_db_connection, upsert_scan_result, create_scan_history, complete_scan_history, get_extracted_fields
    from services.rag import hybrid_retrieve
    from services.ai_engine import score_candidate
    from services.utils import safe_json

    try:
        resumes = list_resumes(client)
        parsed_resumes = [r for r in resumes if r.get("status") == "parsed"]

        if not parsed_resumes:
            st.warning("No parsed resumes available. Upload and parse resumes first.")
            return

        role_reqs = safe_json(role.get("requirements"), [])
        scan_type = "batch_rescan" if rescan else "manual"
        scan_hist = create_scan_history(
            client, role["id"], len(parsed_resumes),
            scan_type=scan_type, triggered_by=user["id"]
        )

        progress = st.progress(0, text="Starting scan...")
        conn = get_db_connection()

        try:
            for idx, resume in enumerate(parsed_resumes):
                name = resume.get("candidate_name") or resume.get("filename", "")
                progress.progress(
                    (idx + 1) / len(parsed_resumes),
                    text=f"Scoring {name} ({idx+1}/{len(parsed_resumes)})"
                )

                evidence = hybrid_retrieve(
                    conn, client, role.get("jd_text", ""),
                    resume["id"], requirements=role_reqs, top_k=10
                )

                # Build structured profile from extracted fields
                candidate_profile = {}
                ext = get_extracted_fields(client, resume["id"])
                if ext:
                    candidate_profile = safe_json(ext.get("fields"), {})
                if resume.get("candidate_name"):
                    candidate_profile["name"] = resume["candidate_name"]

                result = score_candidate(
                    jd_text=role.get("jd_text", ""),
                    requirements=role_reqs,
                    evidence_chunks=evidence,
                    candidate_profile=candidate_profile,
                )

                upsert_scan_result(client, {
                    "resume_id": resume["id"],
                    "role_id": role["id"],
                    "score": result.get("score", 5),
                    "summary": result.get("summary", ""),
                    "evidence": evidence[:10],
                    "requirement_scores": result.get("requirement_scores", []),
                    "confidence": result.get("confidence", "medium"),
                    "flagged_for_review": result.get("flagged_for_review", False),
                    "scan_history_id": scan_hist["id"],
                })
        finally:
            conn.close()

        complete_scan_history(client, scan_hist["id"])
        progress.progress(1.0, text="Scan complete")
        st.success(f'Scanned {len(parsed_resumes)} resumes against "{role["title"]}"')
        st.rerun()

    except Exception as e:
        st.error(f"Scan failed: {str(e)}")

try:
    from services.database import (
        get_supabase_client, list_roles, list_resumes,
        get_scan_results_for_role, get_db_connection,
        upsert_scan_result, create_scan_history, complete_scan_history,
        get_feedback, save_feedback, get_or_create_default_user,
        get_duplicates_for_resume,
    )
    from services.rag import hybrid_retrieve
    from services.ai_engine import score_candidate
    from services.utils import safe_json, confidence_label

    client = get_supabase_client()
    user = get_or_create_default_user(client)

    # Step 1: Select a job
    roles = list_roles(client, status="active")

    if not roles:
        st.caption("No active jobs. Create a job first from the Jobs page.")
        st.stop()

    role_options = {r["title"]: r for r in roles}
    selected_role_title = st.selectbox("Select Job", options=list(role_options.keys()))
    selected_role = role_options[selected_role_title]
    selected_role_id = selected_role["id"]
    st.session_state.current_role_id = selected_role_id

    # Step 2: Scan action
    scan_col1, scan_col2 = st.columns([1, 4])
    with scan_col1:
        if st.button("Scan All Resumes", type="primary"):
            _do_scan(client, selected_role, user)
    with scan_col2:
        if st.button("Re-Scan All"):
            _do_scan(client, selected_role, user, rescan=True)

    st.markdown("---")

    # Step 3: View results
    results = get_scan_results_for_role(client, selected_role_id)

    if not results:
        st.caption("No candidates scanned for this job yet. Click 'Scan All Resumes' above.")
        st.stop()

    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        min_score = st.slider("Min Score", 0.0, 10.0, 0.0, 0.5)
    with filter_col2:
        confidence_filter = st.multiselect(
            "Confidence", ["high", "medium", "low"],
            default=["high", "medium", "low"]
        )
    with filter_col3:
        sort_by = st.selectbox("Sort", ["Score (High to Low)", "Score (Low to High)", "Name"])

    # Filter with proper None handling
    filtered = []
    for r in results:
        score = r.get("score")
        if score is None:
            score = 0.0
        conf = r.get("confidence") or "medium"
        if score >= min_score and conf in confidence_filter:
            filtered.append(r)

    if sort_by == "Score (High to Low)":
        filtered.sort(key=lambda x: x.get("score") or 0, reverse=True)
    elif sort_by == "Score (Low to High)":
        filtered.sort(key=lambda x: x.get("score") or 0)
    else:
        filtered.sort(key=lambda x: ((x.get("resumes") or {}).get("candidate_name") or "").lower())

    st.caption(f"Showing {len(filtered)} of {len(results)} candidates")

    # Candidate list
    for scan in filtered:
        try:
            resume_info = scan.get("resumes") or {}
            candidate_name = resume_info.get("candidate_name") or resume_info.get("filename", "Unknown")
            score = float(scan.get("score") or 0)
            conf = scan.get("confidence") or "medium"
            flagged = bool(scan.get("flagged_for_review", False))
            summary = scan.get("summary") or ""
            resume_id = scan.get("resume_id")

            if not resume_id:
                continue

            duplicates = get_duplicates_for_resume(client, resume_id)
            feedback = get_feedback(client, resume_id, selected_role_id)

            with st.container():
                header_col1, header_col2, header_col3, header_col4 = st.columns([4, 1.5, 1.5, 1])

                with header_col1:
                    name_parts = [candidate_name]
                    if flagged:
                        name_parts.append("[Flagged]")
                    if duplicates:
                        name_parts.append("[Duplicate]")
                    if feedback and feedback.get("decision"):
                        name_parts.append(f"[{feedback['decision'].title()}]")
                    st.markdown(f"#### {' '.join(name_parts)}")

                with header_col2:
                    st.markdown(f"**{score:.1f}**/10")

                with header_col3:
                    st.markdown(f"{confidence_label(conf)} confidence")

                with header_col4:
                    if st.button("View Detail", key=f"detail_{resume_id}"):
                        st.session_state.current_resume_id = resume_id
                        st.session_state.current_role_id = selected_role_id
                        st.switch_page("pages/5_Candidate_Detail.py")

                if summary:
                    st.caption(summary)

                btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1, 5])

                current_decision = feedback.get("decision") if feedback else None

                with btn_col1:
                    if st.button("Shortlist", key=f"sl_{resume_id}", disabled=bool(current_decision == "shortlist")):
                        save_feedback(client, resume_id, selected_role_id, "shortlist", decided_by=user["id"])
                        st.rerun()
                with btn_col2:
                    if st.button("Reject", key=f"rj_{resume_id}", disabled=bool(current_decision == "reject")):
                        save_feedback(client, resume_id, selected_role_id, "reject", decided_by=user["id"])
                        st.rerun()
                with btn_col3:
                    if st.button("Maybe", key=f"mb_{resume_id}", disabled=bool(current_decision == "maybe")):
                        save_feedback(client, resume_id, selected_role_id, "maybe", decided_by=user["id"])
                        st.rerun()

                st.markdown("---")

        except Exception as candidate_err:
            st.warning(f"Could not display candidate: {str(candidate_err)}")
            st.markdown("---")

    # Summary stats
    st.markdown("### Summary")
    stat_col1, stat_col2, stat_col3 = st.columns(3)

    scores = [float(r.get("score") or 0) for r in filtered]
    with stat_col1:
        st.metric("Score 7+", int(len([s for s in scores if s >= 7])))
    with stat_col2:
        st.metric("Flagged", int(len([r for r in filtered if r.get("flagged_for_review")])))
    with stat_col3:
        feedbacks = [get_feedback(client, r.get("resume_id"), selected_role_id) for r in filtered if r.get("resume_id")]
        decided = int(len([f for f in feedbacks if f]))
        st.metric("Reviewed", f"{decided}/{len(filtered)}")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.caption("Check configuration and ensure candidates have been scanned.")

