"""
Dashboard: HR overview with key metrics and quick actions.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard", page_icon="R", layout="wide")

st.markdown("# Dashboard")

try:
    from services.database import get_supabase_client, list_roles, list_resumes

    client = get_supabase_client()

    roles = list_roles(client)
    resumes = list_resumes(client)

    active_roles = [r for r in roles if r.get("status") == "active"]
    parsed_resumes = [r for r in resumes if r.get("status") == "parsed"]

    # Count shortlisted and awaiting review
    feedback_list = client.table("recruiter_feedback").select("resume_id, role_id, decision").execute().data
    shortlisted = len([f for f in feedback_list if f.get("decision") == "shortlist"])

    all_scans = client.table("scan_results").select("resume_id, role_id").execute().data
    scan_keys = {(s["resume_id"], s["role_id"]) for s in all_scans}
    feedback_keys = {(f["resume_id"], f["role_id"]) for f in feedback_list}
    awaiting_review = len(scan_keys - feedback_keys)

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Open Jobs", len(active_roles))
    with col2:
        st.metric("Parsed Resumes", len(parsed_resumes))
    with col3:
        st.metric("Awaiting Review", max(0, awaiting_review))
    with col4:
        st.metric("Shortlisted", shortlisted)

    st.markdown("---")

    # Recent activity
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Recent Jobs")
        if roles:
            df_roles = pd.DataFrame([{
                "Title": r.get("title", ""),
                "Department": r.get("department", "N/A"),
                "Status": r.get("status", "").title(),
                "Created": r.get("created_at", "")[:10],
            } for r in roles[:5]])
            st.dataframe(df_roles, use_container_width=True, hide_index=True)
        else:
            st.caption("No jobs created yet.")

    with col_right:
        st.markdown("### Recent Resumes")
        if resumes:
            df_resumes = pd.DataFrame([{
                "Candidate": r.get("candidate_name") or "Unknown",
                "File": r.get("filename", ""),
                "Status": r.get("status", "").title(),
                "Uploaded": r.get("created_at", "")[:10],
            } for r in resumes[:5]])
            st.dataframe(df_resumes, use_container_width=True, hide_index=True)
        else:
            st.caption("No resumes uploaded yet.")

    # Quick actions
    st.markdown("---")
    st.markdown("### Quick Actions")
    act_col1, act_col2, act_col3 = st.columns(3)
    with act_col1:
        if st.button("Upload Resumes", use_container_width=True):
            st.switch_page("pages/3_Resumes.py")
    with act_col2:
        if st.button("Manage Jobs", use_container_width=True):
            st.switch_page("pages/2_Jobs.py")
    with act_col3:
        if st.button("Review Candidates", use_container_width=True):
            st.switch_page("pages/4_Screening.py")

except Exception as e:
    st.error(f"Could not load dashboard: {str(e)}")
    st.caption("Check that your .env file is configured and the database schema is set up.")
