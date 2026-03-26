"""
Multi-Role Match: compare candidates across roles and roles across candidates.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Multi-Role Match", page_icon="R", layout="wide")

st.markdown("# Multi-Role Match")

try:
    from services.database import (
        get_supabase_client, list_resumes, list_roles,
        get_scan_result, get_scan_results_for_role,
        get_db_connection, upsert_scan_result,
        get_or_create_default_user, get_extracted_fields,
    )
    from services.rag import hybrid_retrieve
    from services.ai_engine import score_candidate
    from services.utils import safe_json, confidence_label

    client = get_supabase_client()
    user = get_or_create_default_user(client)

    tab_candidate_to_roles, tab_role_to_candidates = st.tabs([
        "Candidate → Multiple Roles",
        "Role → Multiple Candidates",
    ])

    # ---- Tab 1: Candidate → Multiple Roles ----
    with tab_candidate_to_roles:
        st.markdown("### Compare a candidate across multiple roles")

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

        selected_candidate = st.selectbox("Candidate", list(resume_options.keys()), key="c2r_candidate")
        resume_id = resume_options[selected_candidate]

        roles = list_roles(client, status="active")
        if not roles:
            st.caption("No active roles available.")
            st.stop()

        role_options = {r["title"]: r for r in roles}
        selected_roles = st.multiselect(
            "Roles to compare",
            list(role_options.keys()),
            default=list(role_options.keys())[:3],
            key="c2r_roles",
        )

        if not selected_roles:
            st.caption("Select at least one role.")
            st.stop()

        if st.button("Score Against Selected Roles", type="primary", key="c2r_score"):
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

                    # Build structured profile from extracted fields
                    candidate_profile = {}
                    ext = get_extracted_fields(client, resume_id)
                    if ext:
                        candidate_profile = safe_json(ext.get("fields"), {})
                    if resume and resume.get("candidate_name"):
                        candidate_profile["name"] = resume["candidate_name"]

                    result = score_candidate(
                        jd_text=role.get("jd_text", ""),
                        requirements=role_reqs,
                        evidence_chunks=evidence,
                        candidate_profile=candidate_profile,
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
                    "Score": f"{scan.get('score', 0):.1f}/10",
                    "Confidence": confidence_label(scan.get("confidence") or "medium"),
                    "Summary": ((scan.get("summary") or "N/A") or "N/A")[:100],
                })
            else:
                comparison_data.append({
                    "Role": role_title,
                    "Department": role.get("department", "N/A"),
                    "Score": "Not scanned",
                    "Confidence": "N/A",
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

            score_val = scan.get("score") or 0
            with st.expander(f"{role_title}: {score_val:.1f}/10"):
                st.markdown(f"**Summary:** {scan.get('summary', 'N/A')}")

                req_scores = safe_json(scan.get("requirement_scores"), [])
                if req_scores:
                    for rs in req_scores:
                        cat = rs.get("category", "must_have").replace("_", " ").title()
                        s = rs.get("score") or 0
                        st.markdown(f"- **{rs.get('requirement', '')}** [{cat}]: {s}/10")
                        st.caption(rs.get("explanation", ""))

                if st.button("View Full Detail", key=f"detail_multi_{role['id']}"):
                    st.session_state.current_resume_id = resume_id
                    st.session_state.current_role_id = role["id"]
                    st.switch_page("pages/5_Candidate_Detail.py")

    # ---- Tab 2: Role → Multiple Candidates ----
    with tab_role_to_candidates:
        st.markdown("### Compare multiple candidates for one role")

        roles_r2c = list_roles(client, status="active")
        if not roles_r2c:
            st.caption("No active roles available.")
            st.stop()

        role_options_r2c = {r["title"]: r for r in roles_r2c}
        selected_role_title = st.selectbox("Role", list(role_options_r2c.keys()), key="r2c_role")
        selected_role = role_options_r2c[selected_role_title]

        resumes_r2c = list_resumes(client)
        parsed_resumes = [r for r in resumes_r2c if r.get("status") == "parsed"]

        if not parsed_resumes:
            st.caption("No parsed resumes available.")
            st.stop()

        candidate_options = {
            f"{r.get('candidate_name') or r['filename']}": r["id"]
            for r in parsed_resumes
        }

        selected_candidates = st.multiselect(
            "Candidates to compare",
            list(candidate_options.keys()),
            default=list(candidate_options.keys())[:5],
            key="r2c_candidates",
        )

        if not selected_candidates:
            st.caption("Select at least one candidate.")
            st.stop()

        if st.button("Score Selected Candidates", type="primary", key="r2c_score"):
            progress = st.progress(0.0, text="Starting comparison...")
            conn = get_db_connection()

            try:
                for idx, cand_name in enumerate(selected_candidates):
                    cand_resume_id = candidate_options[cand_name]
                    progress.progress(
                        (idx + 1) / len(selected_candidates),
                        text=f"Scoring {cand_name}..."
                    )

                    existing = get_scan_result(client, cand_resume_id, selected_role["id"])
                    if existing:
                        continue

                    resume = next((r for r in parsed_resumes if r["id"] == cand_resume_id), None)
                    if not resume:
                        continue

                    role_reqs = safe_json(selected_role.get("requirements"), [])
                    evidence = hybrid_retrieve(
                        conn, client, selected_role.get("jd_text", ""),
                        cand_resume_id, requirements=role_reqs, top_k=10
                    )

                    # Build structured profile from extracted fields
                    candidate_profile = {}
                    ext = get_extracted_fields(client, cand_resume_id)
                    if ext:
                        candidate_profile = safe_json(ext.get("fields"), {})
                    if resume and resume.get("candidate_name"):
                        candidate_profile["name"] = resume["candidate_name"]

                    result = score_candidate(
                        jd_text=selected_role.get("jd_text", ""),
                        requirements=role_reqs,
                        evidence_chunks=evidence,
                        candidate_profile=candidate_profile,
                    )

                    upsert_scan_result(client, {
                        "resume_id": cand_resume_id,
                        "role_id": selected_role["id"],
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
        st.markdown("### Candidate Comparison")

        comparison_data_r2c = []
        for cand_name in selected_candidates:
            cand_resume_id = candidate_options[cand_name]
            scan = get_scan_result(client, cand_resume_id, selected_role["id"])

            if scan:
                score_val = scan.get("score") or 0
                comparison_data_r2c.append({
                    "Candidate": cand_name,
                    "Score": f"{score_val:.1f}/10",
                    "Confidence": confidence_label(scan.get("confidence") or "medium"),
                    "Flagged": "Yes" if scan.get("flagged_for_review") else "No",
                    "Summary": ((scan.get("summary") or "N/A") or "N/A")[:100],
                })
            else:
                comparison_data_r2c.append({
                    "Candidate": cand_name,
                    "Score": "Not scanned",
                    "Confidence": "N/A",
                    "Flagged": "N/A",
                    "Summary": "Click Score to generate.",
                })

        if comparison_data_r2c:
            # Sort by score (scanned first, then by score desc)
            comparison_data_r2c.sort(
                key=lambda x: float(x["Score"].replace("/10", "")) if x["Score"] != "Not scanned" else -1,
                reverse=True
            )
            df_r2c = pd.DataFrame(comparison_data_r2c)
            st.dataframe(df_r2c, use_container_width=True, hide_index=True)

        # Detailed view
        st.markdown("---")
        st.markdown("### Detailed Breakdown")

        for cand_name in selected_candidates:
            cand_resume_id = candidate_options[cand_name]
            scan = get_scan_result(client, cand_resume_id, selected_role["id"])

            if not scan:
                continue

            score_val = scan.get("score") or 0
            with st.expander(f"{cand_name}: {score_val:.1f}/10"):
                st.markdown(f"**Summary:** {scan.get('summary', 'N/A')}")

                req_scores = safe_json(scan.get("requirement_scores"), [])
                if req_scores:
                    for rs in req_scores:
                        cat = rs.get("category", "must_have").replace("_", " ").title()
                        s = rs.get("score") or 0
                        st.markdown(f"- **{rs.get('requirement', '')}** [{cat}]: {s}/10")
                        st.caption(rs.get("explanation", ""))

                if st.button("View Full Detail", key=f"detail_r2c_{cand_resume_id}"):
                    st.session_state.current_resume_id = cand_resume_id
                    st.session_state.current_role_id = selected_role["id"]
                    st.switch_page("pages/5_Candidate_Detail.py")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.caption("Check your .env configuration.")
