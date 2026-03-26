"""
Jobs: CRM-style job description management with scanning.
"""

import streamlit as st
import json

st.set_page_config(page_title="Jobs", page_icon="R", layout="wide")

st.markdown("# Jobs")


def _run_scan(client, role, user, rescan=False):
    from services.database import list_resumes, get_db_connection, upsert_scan_result, create_scan_history, complete_scan_history
    from services.rag import hybrid_retrieve
    from services.ai_engine import score_candidate
    from services.utils import safe_json

    try:
        resumes = list_resumes(client)
        parsed_resumes = [r for r in resumes if r.get("status") == "parsed"]

        if not parsed_resumes:
            st.warning("No parsed resumes available to scan.")
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

                result = score_candidate(
                    jd_text=role.get("jd_text", ""),
                    requirements=role_reqs,
                    evidence_chunks=evidence,
                    full_resume_text=resume.get("raw_text", ""),
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
        get_supabase_client, list_roles, create_role, update_role,
        delete_role, get_or_create_default_user,
        list_resumes, get_scan_results_for_role,
        get_db_connection, upsert_scan_result, create_scan_history,
        complete_scan_history, get_default_config,
    )
    from services.rag import hybrid_retrieve
    from services.ai_engine import score_candidate
    from services.utils import safe_json

    client = get_supabase_client()
    user = get_or_create_default_user(client)

    # Delete confirmation state
    if "delete_role_id" not in st.session_state:
        st.session_state.delete_role_id = None

    # Add New Job toggle
    if "show_add_job" not in st.session_state:
        st.session_state.show_add_job = False

    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        roles = list_roles(client)
        active_count = len([r for r in roles if r.get("status") == "active"])
        st.caption(f"{len(roles)} total jobs, {active_count} active")
    with header_col2:
        if st.button("Add New Job", use_container_width=True):
            st.session_state.show_add_job = not st.session_state.show_add_job
            st.rerun()

    # Add New Job form
    if st.session_state.show_add_job:
        st.markdown("---")
        st.markdown("### New Job")

        role_title = st.text_input("Job Title *", placeholder="Senior Backend Engineer")
        department = st.text_input("Department", placeholder="Engineering")
        jd_text = st.text_area("Job Description *", height=150, placeholder="Paste the full job description...")

        st.markdown("**Requirements**")
        st.caption("Categorize as Must Have (3x weight), Good to Have (2x), or Bonus (1x).")

        if "new_requirements" not in st.session_state:
            st.session_state.new_requirements = [{"text": "", "category": "must_have"}]

        reqs = st.session_state.new_requirements

        for i, req in enumerate(reqs):
            col1, col2, col3 = st.columns([5, 2, 1])
            with col1:
                reqs[i]["text"] = st.text_input(
                    f"Requirement {i+1}", value=req["text"], key=f"req_text_{i}",
                    label_visibility="collapsed", placeholder="e.g., 5+ years Python experience",
                )
            with col2:
                reqs[i]["category"] = st.selectbox(
                    f"Category {i+1}",
                    ["must_have", "good_to_have", "bonus"],
                    index=["must_have", "good_to_have", "bonus"].index(req.get("category", "must_have")),
                    key=f"req_cat_{i}", label_visibility="collapsed",
                    format_func=lambda x: x.replace("_", " ").title(),
                )
            with col3:
                if i > 0 and st.button("Remove", key=f"req_del_{i}"):
                    reqs.pop(i)
                    st.rerun()

        if st.button("Add Requirement"):
            reqs.append({"text": "", "category": "must_have"})
            st.rerun()

        create_col1, create_col2 = st.columns([1, 4])
        with create_col1:
            if st.button("Create Job", type="primary"):
                if not role_title or not jd_text:
                    st.error("Job title and job description are required.")
                else:
                    valid_reqs = [r for r in reqs if r["text"].strip()]
                    try:
                        create_role(
                            client, title=role_title, department=department,
                            jd_text=jd_text, requirements=valid_reqs, created_by=user["id"],
                        )
                        st.success(f'Job "{role_title}" created.')
                        st.session_state.new_requirements = [{"text": "", "category": "must_have"}]
                        st.session_state.show_add_job = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create job: {str(e)}")
        with create_col2:
            if st.button("Cancel"):
                st.session_state.show_add_job = False
                st.rerun()

    # Existing Jobs
    st.markdown("---")

    if not roles:
        st.caption("No jobs created yet. Click 'Add New Job' to get started.")
    else:
        for role in roles:
            role_reqs = safe_json(role.get("requirements"), [])
            is_active = role.get("status") == "active"
            status_text = "Active" if is_active else "Archived"
            scans = get_scan_results_for_role(client, role["id"])

            # Job card header
            card_col1, card_col2, card_col3, card_col4 = st.columns([3, 1.5, 1, 1.5])

            with card_col1:
                st.markdown(f"#### {role['title']}")
                dept = role.get("department", "")
                meta_parts = []
                if dept:
                    meta_parts.append(dept)
                meta_parts.append(f"{len(role_reqs)} requirements")
                if scans:
                    meta_parts.append(f"{len(scans)} candidates scanned")
                st.caption(" | ".join(meta_parts))

            with card_col2:
                st.caption(f"Created: {role.get('created_at', '')[:10]}")

            with card_col3:
                if is_active:
                    st.markdown(f'<span style="color: #6ee7b7; font-weight: 600;">{status_text}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span style="color: #9ca3af; font-weight: 600;">{status_text}</span>', unsafe_allow_html=True)

            with card_col4:
                pass

            # Expandable details
            with st.expander("View Details"):
                jd = role.get("jd_text", "")
                st.markdown("**Job Description:**")
                st.markdown(jd if len(jd) <= 500 else jd[:500] + "...")

                if role_reqs:
                    st.markdown("**Requirements:**")
                    for req in role_reqs:
                        cat = req.get("category", "must_have").replace("_", " ").title()
                        st.markdown(f"- **[{cat}]** {req.get('text', '')}")

            # Actions row
            act_col1, act_col2, act_col3, act_col4 = st.columns(4)

            with act_col1:
                if st.button("Scan All Resumes", key=f"scan_{role['id']}"):
                    _run_scan(client, role, user)

            with act_col2:
                if st.button("Re-Scan", key=f"rescan_{role['id']}"):
                    _run_scan(client, role, user, rescan=True)

            with act_col3:
                new_status = "archived" if is_active else "active"
                label = "Archive" if is_active else "Activate"
                if st.button(label, key=f"status_{role['id']}"):
                    update_role(client, role["id"], {"status": new_status})
                    st.rerun()

            with act_col4:
                if st.session_state.delete_role_id == role["id"]:
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Confirm", key=f"confirm_del_role_{role['id']}", type="primary"):
                            delete_role(client, role["id"])
                            st.session_state.delete_role_id = None
                            st.rerun()
                    with c2:
                        if st.button("Cancel", key=f"cancel_del_role_{role['id']}"):
                            st.session_state.delete_role_id = None
                            st.rerun()
                else:
                    if st.button("Delete", key=f"del_{role['id']}"):
                        st.session_state.delete_role_id = role["id"]
                        st.rerun()

            st.markdown("---")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.caption("Check your .env configuration.")

