"""
Scan History: audit trail of all scan runs.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Scan History", page_icon="R", layout="wide")

st.markdown("# Scan History")

try:
    from services.database import get_supabase_client, list_scan_history, list_roles

    client = get_supabase_client()

    # Filters
    filter_col1, filter_col2 = st.columns(2)

    roles = list_roles(client)
    role_map = {r["title"]: r["id"] for r in roles}

    with filter_col1:
        role_filter = st.selectbox("Role", ["All"] + list(role_map.keys()))
    with filter_col2:
        type_filter = st.selectbox("Type", ["All", "manual", "batch_rescan", "auto"])

    role_id = role_map.get(role_filter) if role_filter != "All" else None
    history = list_scan_history(client, role_id=role_id)

    if type_filter != "All":
        history = [h for h in history if h.get("scan_type") == type_filter]

    if not history:
        # Fallback: show scan_results summary when scan_history table is empty
        scan_results_all = client.table("scan_results").select("role_id, score, created_at, roles(title)").order("created_at", desc=True).execute().data

        if not scan_results_all:
            st.caption("No scan history found.")
            st.stop()

        st.caption("Showing scan results summary (detailed scan history will appear for future scans).")

        role_groups = {}
        for sr in scan_results_all:
            rid = sr.get("role_id")
            if rid not in role_groups:
                role_groups[rid] = {
                    "Role": sr.get("roles", {}).get("title", "N/A") if sr.get("roles") else "N/A",
                    "Candidates": 0,
                    "Avg Score": 0,
                    "Latest": sr.get("created_at", "")[:10],
                }
            role_groups[rid]["Candidates"] += 1
            role_groups[rid]["Avg Score"] += sr.get("score", 0)
        for rg in role_groups.values():
            if rg["Candidates"] > 0:
                rg["Avg Score"] = round(rg["Avg Score"] / rg["Candidates"], 1)

        df_fallback = pd.DataFrame(list(role_groups.values()))
        st.dataframe(df_fallback, use_container_width=True, hide_index=True)

        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric("Total Scanned", len(scan_results_all))
        with stat_col2:
            avg = sum(s["score"] for s in scan_results_all) / len(scan_results_all)
            st.metric("Overall Avg Score", f"{avg:.1f}")

        st.stop()

    # History table
    st.caption(f"{len(history)} scan(s)")

    df = pd.DataFrame([{
        "Role": h.get("roles", {}).get("title", "N/A") if h.get("roles") else "N/A",
        "Resumes": h.get("resume_count", 0),
        "Type": h.get("scan_type", "N/A").replace("_", " ").title(),
        "Status": h.get("status", "N/A").title(),
        "Started": h.get("started_at", "")[:16].replace("T", " "),
        "Completed": (h.get("completed_at") or "N/A")[:16].replace("T", " ") if h.get("completed_at") else "N/A",
    } for h in history])

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Stats
    st.markdown("---")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("Total Scans", len(history))
    with stat_col2:
        st.metric("Resumes Scanned", sum(h.get("resume_count", 0) for h in history))
    with stat_col3:
        st.metric("Completed", len([h for h in history if h.get("status") == "completed"]))
    with stat_col4:
        st.metric("Failed", len([h for h in history if h.get("status") == "failed"]))

    # Volume chart
    if len(history) > 2:
        st.markdown("---")
        st.markdown("### Scan Volume")
        import plotly.express as px

        for h in history:
            h["date"] = h.get("started_at", "")[:10]

        dates = pd.DataFrame(history)
        if "date" in dates.columns:
            volume = dates.groupby("date").size().reset_index(name="scans")
            fig = px.bar(volume, x="date", y="scans",
                         labels={"date": "Date", "scans": "Scans"},
                         color_discrete_sequence=["#3b82f6"])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#9ca3af",
                margin=dict(l=0, r=0, t=10, b=0),
                height=250,
            )
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.caption("Check your .env configuration.")
