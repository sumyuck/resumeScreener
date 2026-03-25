"""
Role Management: create, edit, and manage job descriptions with weighted requirements.
"""

import streamlit as st
import json

st.set_page_config(page_title="Role Management", page_icon="R", layout="wide")

st.markdown("# Role Management")

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

    tab_create, tab_manage = st.tabs(["Create Role", "Existing Roles"])

    # Create Role
    with tab_create:
        role_title = st.text_input("Role Title *", placeholder="Senior Backend Engineer")
        department = st.text_input("Department", placeholder="Engineering")
        jd_text = st.text_area("Job Description *", height=180, placeholder="Paste the full job description...")

        st.markdown("#### Requirements")
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

        st.markdown("---")

        if st.button("Create Role", type="primary"):
            if not role_title or not jd_text:
                st.error("Role title and job description are required.")
            else:
                valid_reqs = [r for r in reqs if r["text"].strip()]
                try:
                    role = create_role(
                        client, title=role_title, department=department,
                        jd_text=jd_text, requirements=valid_reqs, created_by=user["id"],
                    )
                    st.success(f"Role \"{role_title}\" created.")
                    st.session_state.new_requirements = [{"text": "", "category": "must_have"}]
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create role: {str(e)}")

    # Manage Existing Roles
    with tab_manage:
        roles = list_roles(client)

        if not roles:
            st.caption("No roles created yet.")
        else:
            for role in roles:
                role_reqs = safe_json(role.get("requirements"), [])
                status_label = "Active" if role.get("status") == "active" else "Archived"

                with st.expander(f"{role['title']} / {role.get('department', 'N/A')} ({status_label})"):
                    # JD preview
                    jd = role.get("jd_text", "")
                    if len(jd) > 400:
                        st.markdown(jd[:400] + "...")
                        with st.popover("Full JD"):
                            st.markdown(jd)
                    else:
                        st.markdown(jd)

                    # Requirements
                    if role_reqs:
                        st.markdown("**Requirements:**")
                        for req in role_reqs:
                            cat = req.get("category", "must_have").replace("_", " ").title()
                            st.markdown(f"- **[{cat}]** {req.get('text', '')}")

                    # Scan summary
                    scans = get_scan_results_for_role(client, role["id"])
                    if scans:
                        avg = sum(s["score"] for s in scans) / len(scans)
                        st.caption(f"{len(scans)} candidates scanned, avg score: {avg:.1f}")

                    # Actions
                    action_col1, action_col2, action_col3, action_col4 = st.columns(4)

                    with action_col1:
                        if st.button("Scan All", key=f"scan_{role['id']}"):
                            _run_scan(client, role, user)

                    with action_col2:
                        if st.button("Re-Scan", key=f"rescan_{role['id']}"):
                            _run_scan(client, role, user, rescan=True)

                    with action_col3:
                        new_status = "archived" if role.get("status") == "active" else "active"
                        label = "Archive" if new_status == "archived" else "Activate"
                        if st.button(label, key=f"status_{role['id']}"):
                            update_role(client, role["id"], {"status": new_status})
                            st.rerun()

                    with action_col4:
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

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.caption("Check your .env configuration.")


def _run_scan(client, role, user, rescan=False):
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
        st.success(f"Scanned {len(parsed_resumes)} resumes against \"{role['title']}\"")
        st.rerun()

    except Exception as e:
        st.error(f"Scan failed: {str(e)}")
