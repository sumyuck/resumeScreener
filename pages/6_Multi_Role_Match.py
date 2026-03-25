"""
Multi-Role Match: compare a single candidate across multiple roles.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Multi-Role Match", page_icon="R", layout="wide")

st.markdown("# Multi-Role Match")

try:
    from services.database import (
        get_supabase_client, list_resumes, list_roles,
        get_scan_result, get_db_connection, upsert_scan_result,
        get_or_create_default_user,
    )
    from services.rag import hybrid_retrieve
    from services.ai_engine import score_candidate
    from services.utils import safe_json, confidence_label

    client = get_supabase_client()
    user = get_or_create_default_user(client)

    # Select candidate
    resumes = list_resumes(client)
    if not resumes:
        st.caption("No resumes uploaded yet.")
        st.stop()

    resume_options = {
        f"{r.get('candidate_name') or r['filename']}": r["id"]
        for r in resumes if r.get("status") == "parsed"
    }

    if not resume_options:
        st.caption("No parsed resumes available.")
        st.stop()

    selected_candidate = st.selectbox("Candidate", list(resume_options.keys()))
    resume_id = resume_options[selected_candidate]

    # Select roles
    roles = list_roles(client, status="active")
    if not roles:
        st.caption("No active roles available.")
        st.stop()

    role_options = {r["title"]: r for r in roles}
    selected_roles = st.multiselect(
        "Roles to compare",
        list(role_options.keys()),
        default=list(role_options.keys())[:3],
    )

    if not selected_roles:
        st.caption("Select at least one role.")
        st.stop()

    # Score button
    if st.button("Score Against Selected Roles", type="primary"):
        progress = st.progress(0.0, text="Starting comparison...")
        conn = get_db_connection()

        resume = next((r for r in resumes if r["id"] == resume_id), None)

        try:
            for idx, role_title in enumerate(selected_roles):
                role = role_options[role_title]
                progress.progress(
                    (idx + 1) / len(selected_roles),
                    text=f"Scoring against {role_title}..."
                )

                existing = get_scan_result(client, resume_id, role["id"])
                if existing:
                    continue

                role_reqs = safe_json(role.get("requirements"), [])
                evidence = hybrid_retrieve(
                    conn, client, role.get("jd_text", ""),
                    resume_id, requirements=role_reqs, top_k=10
                )

                result = score_candidate(
                    jd_text=role.get("jd_text", ""),
                    requirements=role_reqs,
                    evidence_chunks=evidence,
                    full_resume_text=resume.get("raw_text", "") if resume else "",
                )

                upsert_scan_result(client, {
                    "resume_id": resume_id,
                    "role_id": role["id"],
                    "score": result.get("score", 5),
                    "summary": result.get("summary", ""),
                    "evidence": evidence[:10],
                    "requirement_scores": result.get("requirement_scores", []),
                    "confidence": result.get("confidence", "medium"),
                    "flagged_for_review": result.get("flagged_for_review", False),
                })
        finally:
            conn.close()

        progress.progress(1.0, text="Complete")
        st.rerun()

    # Comparison table
    st.markdown("---")
    st.markdown("### Score Comparison")

    comparison_data = []
    for role_title in selected_roles:
        role = role_options[role_title]
        scan = get_scan_result(client, resume_id, role["id"])

        if scan:
            comparison_data.append({
                "Role": role_title,
                "Department": role.get("department", "N/A"),
                "Score": f"{scan['score']:.1f}/10",
                "Confidence": confidence_label(scan.get("confidence", "medium")),
                "Flagged": "Yes" if scan.get("flagged_for_review") else "No",
                "Summary": (scan.get("summary", "N/A") or "N/A")[:100],
            })
        else:
            comparison_data.append({
                "Role": role_title,
                "Department": role.get("department", "N/A"),
                "Score": "Not scanned",
                "Confidence": "N/A",
                "Flagged": "N/A",
                "Summary": "Click Score to generate.",
            })

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Detailed breakdown
    st.markdown("---")
    st.markdown("### Detailed Breakdown")

    for role_title in selected_roles:
        role = role_options[role_title]
        scan = get_scan_result(client, resume_id, role["id"])

        if not scan:
            continue

        with st.expander(f"{role_title}: {scan['score']:.1f}/10"):
            st.markdown(f"**Summary:** {scan.get('summary', 'N/A')}")

            req_scores = safe_json(scan.get("requirement_scores"), [])
            if req_scores:
                for rs in req_scores:
                    cat = rs.get("category", "must_have").replace("_", " ").title()
                    s = rs.get("score", 0)
                    st.markdown(f"- **{rs.get('requirement', '')}** [{cat}]: {s}/10")
                    st.caption(rs.get("explanation", ""))

            if st.button("View Full Detail", key=f"detail_multi_{role['id']}"):
                st.session_state.current_resume_id = resume_id
                st.session_state.current_role_id = role["id"]
                st.switch_page("pages/5_Candidate_Detail.py")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.caption("Check your .env configuration.")
