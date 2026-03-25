"""
Candidate Review: sortable, filterable candidate ranking per role.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Candidate Review", page_icon="R", layout="wide")

st.markdown("# Candidate Review")

try:
    from services.database import (
        get_supabase_client, list_roles, get_scan_results_for_role,
        get_feedback, save_feedback, get_or_create_default_user,
        get_duplicates_for_resume,
    )
    from services.utils import safe_json, confidence_label

    client = get_supabase_client()
    user = get_or_create_default_user(client)

    # Role selector
    roles = list_roles(client, status="active")

    if not roles:
        st.caption("No active roles. Create a role first.")
        st.stop()

    role_options = {r["title"]: r["id"] for r in roles}
    selected_role_title = st.selectbox("Role", options=list(role_options.keys()))
    selected_role_id = role_options[selected_role_title]
    st.session_state.current_role_id = selected_role_id

    results = get_scan_results_for_role(client, selected_role_id)

    if not results:
        st.caption("No candidates scanned for this role. Run a scan from Role Management.")
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

    filtered = [
        r for r in results
        if r["score"] >= min_score and r.get("confidence", "medium") in confidence_filter
    ]

    if sort_by == "Score (High to Low)":
        filtered.sort(key=lambda x: x["score"], reverse=True)
    elif sort_by == "Score (Low to High)":
        filtered.sort(key=lambda x: x["score"])
    else:
        filtered.sort(key=lambda x: (x.get("resumes", {}) or {}).get("candidate_name", "") or "")

    st.caption(f"Showing {len(filtered)} of {len(results)} candidates")
    st.markdown("---")

    # Candidate list
    for scan in filtered:
        resume_info = scan.get("resumes") or {}
        candidate_name = resume_info.get("candidate_name") or resume_info.get("filename", "Unknown")
        score = scan["score"]
        conf = scan.get("confidence", "medium")
        flagged = scan.get("flagged_for_review", False)
        summary = scan.get("summary", "")
        resume_id = scan.get("resume_id")

        # Duplicate check
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
                if feedback:
                    name_parts.append(f"[{feedback['decision'].title()}]")
                st.markdown(f"#### {' '.join(name_parts)}")

            with header_col2:
                st.markdown(f"**{score:.1f}**/10")

            with header_col3:
                st.markdown(f"{confidence_label(conf)} confidence")

            with header_col4:
                if st.button("Detail", key=f"detail_{resume_id}"):
                    st.session_state.current_resume_id = resume_id
                    st.switch_page("pages/5_Candidate_Detail.py")

            if summary:
                st.caption(summary)

            btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([1, 1, 1, 5])

            with btn_col1:
                if st.button("Shortlist", key=f"sl_{resume_id}", disabled=feedback and feedback["decision"] == "shortlist"):
                    save_feedback(client, resume_id, selected_role_id, "shortlist", decided_by=user["id"])
                    st.rerun()
            with btn_col2:
                if st.button("Reject", key=f"rj_{resume_id}", disabled=feedback and feedback["decision"] == "reject"):
                    save_feedback(client, resume_id, selected_role_id, "reject", decided_by=user["id"])
                    st.rerun()
            with btn_col3:
                if st.button("Maybe", key=f"mb_{resume_id}", disabled=feedback and feedback["decision"] == "maybe"):
                    save_feedback(client, resume_id, selected_role_id, "maybe", decided_by=user["id"])
                    st.rerun()

            st.markdown("---")

    # Summary stats
    st.markdown("### Summary")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    scores = [r["score"] for r in filtered]
    with stat_col1:
        st.metric("Avg Score", f"{sum(scores)/len(scores):.1f}" if scores else "N/A")
    with stat_col2:
        st.metric("Score 7+", len([s for s in scores if s >= 7]))
    with stat_col3:
        st.metric("Flagged", len([r for r in filtered if r.get("flagged_for_review")]))
    with stat_col4:
        feedbacks = [get_feedback(client, r["resume_id"], selected_role_id) for r in filtered]
        decided = len([f for f in feedbacks if f])
        st.metric("Reviewed", f"{decided}/{len(filtered)}")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.caption("Check configuration and ensure candidates have been scanned.")
