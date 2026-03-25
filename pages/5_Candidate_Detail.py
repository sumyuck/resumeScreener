"""
Candidate Detail: profile, scores, evidence, interview questions, raw resume.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Candidate Detail", page_icon="R", layout="wide")

st.markdown("# Candidate Detail")

try:
    from services.database import (
        get_supabase_client, get_resume, get_extracted_fields,
        get_scan_results_for_resume, get_scan_result,
        get_feedback, save_feedback, get_or_create_default_user,
        get_duplicates_for_resume, list_roles, get_chunks, list_resumes,
    )
    from services.ai_engine import generate_interview_questions, generate_candidate_summary
    from services.utils import safe_json, confidence_label

    client = get_supabase_client()
    user = get_or_create_default_user(client)

    # Resume selector
    resumes = list_resumes(client)

    if not resumes:
        st.caption("No resumes uploaded yet.")
        st.stop()

    preselected = st.session_state.get("current_resume_id")

    resume_options = {
        f"{r.get('candidate_name') or r['filename']}": r["id"]
        for r in resumes
    }

    default_idx = 0
    if preselected:
        for idx, (label, rid) in enumerate(resume_options.items()):
            if rid == preselected:
                default_idx = idx
                break

    selected_label = st.selectbox("Candidate", list(resume_options.keys()), index=default_idx)
    resume_id = resume_options[selected_label]
    resume = get_resume(client, resume_id)

    if not resume:
        st.error("Resume not found.")
        st.stop()

    # Header
    st.markdown("---")
    head_col1, head_col2, head_col3 = st.columns([3, 1.5, 1.5])

    with head_col1:
        st.markdown(f"## {resume.get('candidate_name') or resume['filename']}")
        if resume.get("candidate_email"):
            st.caption(resume["candidate_email"])
        st.caption(f"{resume['filename']} | {resume.get('file_type', '').upper()} | {resume.get('created_at', '')[:10]}")

    with head_col2:
        quality = resume.get("quality_score")
        st.metric("Quality", f"{quality:.1f}/10" if quality is not None else "N/A")

    with head_col3:
        dups = get_duplicates_for_resume(client, resume_id)
        if dups:
            st.warning(f"{len(dups)} duplicate(s) detected")
        else:
            st.caption("No duplicates")

    st.markdown("---")

    # Tabs
    tab_profile, tab_scores, tab_evidence, tab_interview, tab_raw = st.tabs([
        "Profile", "Scores", "Evidence", "Interview Prep", "Raw Resume"
    ])

    # Profile tab
    with tab_profile:
        ext = get_extracted_fields(client, resume_id)

        if ext:
            fields = safe_json(ext.get("fields"), {})
            for key, value in fields.items():
                label = key.replace("_", " ").title()
                if isinstance(value, list):
                    st.markdown(f"**{label}:** {', '.join(str(v) for v in value)}")
                elif value is not None:
                    st.markdown(f"**{label}:** {value}")
                else:
                    st.markdown(f"**{label}:** Not found")
        else:
            st.caption("No extracted fields. Re-process this resume to extract fields.")

        if resume.get("quality_notes"):
            st.markdown("### Quality Notes")
            st.caption(resume["quality_notes"])

    # Scores tab
    with tab_scores:
        scan_results = get_scan_results_for_resume(client, resume_id)

        if not scan_results:
            st.caption("Not scanned against any roles yet.")
        else:
            for scan in scan_results:
                role_info = scan.get("roles") or {}
                role_title = role_info.get("title", "Unknown Role")
                score = scan["score"]
                conf = scan.get("confidence", "medium")
                flagged = scan.get("flagged_for_review", False)

                flag_text = " [Flagged]" if flagged else ""
                with st.expander(
                    f"{role_title}: {score:.1f}/10, {confidence_label(conf)} confidence{flag_text}",
                    expanded=len(scan_results) == 1
                ):
                    st.markdown(f"**Summary:** {scan.get('summary', 'N/A')}")

                    req_scores = safe_json(scan.get("requirement_scores"), [])
                    if req_scores:
                        st.markdown("#### Requirement Breakdown")
                        for rs in req_scores:
                            cat = rs.get("category", "must_have").replace("_", " ").title()
                            req_score = rs.get("score", 0)
                            st.markdown(f"**{rs.get('requirement', 'N/A')}** [{cat}]: {req_score}/10")
                            st.caption(rs.get("explanation", ""))
                            if rs.get("evidence_snippet"):
                                st.markdown(f"> {rs['evidence_snippet'][:200]}")

                    # Feedback
                    role_id = role_info.get("id") or scan.get("role_id")
                    feedback = get_feedback(client, resume_id, role_id) if role_id else None

                    st.markdown("#### Decision")
                    fb_col1, fb_col2, fb_col3, fb_col4 = st.columns(4)
                    current_decision = feedback["decision"] if feedback else None

                    with fb_col1:
                        if st.button("Shortlist", key=f"fb_sl_{role_id}", disabled=current_decision == "shortlist"):
                            save_feedback(client, resume_id, role_id, "shortlist", decided_by=user["id"])
                            st.rerun()
                    with fb_col2:
                        if st.button("Reject", key=f"fb_rj_{role_id}", disabled=current_decision == "reject"):
                            save_feedback(client, resume_id, role_id, "reject", decided_by=user["id"])
                            st.rerun()
                    with fb_col3:
                        if st.button("Maybe", key=f"fb_mb_{role_id}", disabled=current_decision == "maybe"):
                            save_feedback(client, resume_id, role_id, "maybe", decided_by=user["id"])
                            st.rerun()
                    with fb_col4:
                        if current_decision:
                            st.markdown(f"**Current:** {current_decision.title()}")

    # Evidence tab
    with tab_evidence:
        scan_results = get_scan_results_for_resume(client, resume_id)

        if not scan_results:
            st.caption("No scoring evidence available.")
        else:
            for scan in scan_results:
                role_info = scan.get("roles") or {}
                role_title = role_info.get("title", "Unknown Role")
                evidence = safe_json(scan.get("evidence"), [])

                if evidence:
                    st.markdown(f"### {role_title}")
                    st.caption(f"{len(evidence)} relevant chunks retrieved")

                    for i, ev in enumerate(evidence):
                        match_type = ev.get("match_type", "semantic").title()
                        section = ev.get("section", "general").title()
                        chunk_text = ev.get("chunk_text") or ev.get("text", "N/A")

                        with st.expander(f"Chunk {i+1}: {match_type} match, {section} section"):
                            st.markdown(chunk_text)
                            if ev.get("matched_skills"):
                                st.caption(f"Matched skills: {', '.join(ev['matched_skills'])}")
                            if ev.get("semantic_score"):
                                st.caption(f"Similarity: {ev['semantic_score']:.3f}")
                else:
                    st.caption(f"No evidence for {role_title}")

    # Interview Prep tab
    with tab_interview:
        scan_results = get_scan_results_for_resume(client, resume_id)

        if not scan_results:
            st.caption("Scan the candidate first to generate interview questions.")
        else:
            role_options_iq = {
                (sr.get("roles") or {}).get("title", "Unknown"): sr
                for sr in scan_results
            }
            selected_role_iq = st.selectbox(
                "Generate questions for:",
                list(role_options_iq.keys()),
                key="iq_role"
            )

            if st.button("Generate Questions", type="primary"):
                scan = role_options_iq[selected_role_iq]
                role_info = scan.get("roles") or {}
                role = None
                if role_info.get("id"):
                    from services.database import get_role
                    role = get_role(client, role_info["id"])

                with st.spinner("Generating questions..."):
                    questions = generate_interview_questions(
                        jd_text=role.get("jd_text", "") if role else "",
                        resume_text=resume.get("raw_text", ""),
                        score_result={
                            "score": scan.get("score"),
                            "requirement_scores": safe_json(scan.get("requirement_scores"), []),
                        }
                    )

                if questions:
                    st.markdown(f"### Interview Questions for {selected_role_iq}")
                    for i, q in enumerate(questions):
                        cat = q.get("category", "general").replace("_", " ").title()
                        st.markdown(f"**{i+1}. {q.get('question', '')}**")
                        st.caption(f"Category: {cat} | {q.get('rationale', '')}")
                        st.markdown("---")
                else:
                    st.warning("Could not generate questions. Try again.")

    # Raw Resume tab
    with tab_raw:
        chunks = get_chunks(client, resume_id)

        if chunks:
            st.markdown(f"### Chunks ({len(chunks)} sections)")
            for chunk in chunks:
                with st.expander(f"Chunk {chunk['chunk_index']+1}: {chunk.get('section', 'general').title()}"):
                    st.text(chunk["chunk_text"])

        if resume.get("raw_text"):
            with st.expander("Full Text", expanded=not chunks):
                st.text(resume["raw_text"])
        else:
            st.caption("No text available.")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.caption("Check your .env configuration.")
