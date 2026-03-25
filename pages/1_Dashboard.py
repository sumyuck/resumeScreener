"""
Dashboard: KPIs, recent activity, and quick actions.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard", page_icon="R", layout="wide")

st.markdown("# Dashboard")

try:
    from services.database import get_supabase_client, list_roles, list_resumes, list_scan_history

    client = get_supabase_client()

    roles = list_roles(client)
    resumes = list_resumes(client)
    scan_history = list_scan_history(client)

    all_scans = client.table("scan_results").select("score").execute().data
    avg_score = sum(s["score"] for s in all_scans) / len(all_scans) if all_scans else 0

    feedback_list = client.table("recruiter_feedback").select("resume_id, role_id").execute().data
    feedback_keys = {(f["resume_id"], f["role_id"]) for f in feedback_list}
    pending_feedback = len(all_scans) - len(feedback_keys)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Resumes", len(resumes))
    with col2:
        st.metric("Active Roles", len([r for r in roles if r.get("status") == "active"]))
    with col3:
        st.metric("Avg Score", f"{avg_score:.1f}/10" if all_scans else "N/A")
    with col4:
        st.metric("Pending Reviews", max(0, pending_feedback))

    st.markdown("---")

    # Recent activity
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Recent Resumes")
        if resumes:
            df = pd.DataFrame([{
                "Candidate": r.get("candidate_name") or "Unknown",
                "File": r.get("filename", ""),
                "Status": r.get("status", ""),
                "Uploaded": r.get("created_at", "")[:10],
            } for r in resumes[:8]])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("No resumes uploaded yet.")

    with col_right:
        st.markdown("### Recent Scans")
        if scan_history:
            df_scans = pd.DataFrame([{
                "Role": s.get("roles", {}).get("title", "N/A") if s.get("roles") else "N/A",
                "Resumes": s.get("resume_count", 0),
                "Status": s.get("status", ""),
                "Date": s.get("started_at", "")[:10],
            } for s in scan_history[:8]])
            st.dataframe(df_scans, use_container_width=True, hide_index=True)
        elif all_scans:
            # Fallback: show scan_results grouped by role when scan_history is empty
            scan_results_with_roles = client.table("scan_results").select("role_id, score, created_at, roles(title)").order("created_at", desc=True).limit(20).execute().data
            role_groups = {}
            for sr in scan_results_with_roles:
                rid = sr.get("role_id")
                if rid not in role_groups:
                    role_groups[rid] = {
                        "Role": sr.get("roles", {}).get("title", "N/A") if sr.get("roles") else "N/A",
                        "Candidates": 0,
                        "Avg Score": 0,
                        "Date": sr.get("created_at", "")[:10],
                    }
                role_groups[rid]["Candidates"] += 1
                role_groups[rid]["Avg Score"] += sr.get("score", 0)
            for rg in role_groups.values():
                if rg["Candidates"] > 0:
                    rg["Avg Score"] = round(rg["Avg Score"] / rg["Candidates"], 1)
            if role_groups:
                df_scans = pd.DataFrame(list(role_groups.values()))
                st.dataframe(df_scans, use_container_width=True, hide_index=True)

    # Quick actions
    st.markdown("---")
    act_col1, act_col2, act_col3 = st.columns(3)
    with act_col1:
        if st.button("Upload Resumes", use_container_width=True):
            st.switch_page("pages/2_Upload_Resumes.py")
    with act_col2:
        if st.button("Manage Roles", use_container_width=True):
            st.switch_page("pages/3_Role_Management.py")
    with act_col3:
        if st.button("Review Candidates", use_container_width=True):
            st.switch_page("pages/4_Candidate_Review.py")

    # Score distribution
    if all_scans and len(all_scans) > 2:
        st.markdown("---")
        st.markdown("### Score Distribution")
        import plotly.express as px
        scores_df = pd.DataFrame(all_scans)
        fig = px.histogram(scores_df, x="score", nbins=10, range_x=[0, 10],
                           labels={"score": "Score", "count": "Candidates"},
                           color_discrete_sequence=["#3b82f6"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#9ca3af",
            xaxis=dict(dtick=1),
            margin=dict(l=0, r=0, t=10, b=0),
            height=250,
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Could not load dashboard: {str(e)}")
    st.caption("Check that your .env file is configured and the database schema is set up.")
