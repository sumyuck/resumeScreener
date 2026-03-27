"""
Settings: extraction configuration management and batch re-processing.
"""

import streamlit as st
import json

st.set_page_config(page_title="Settings", page_icon="R", layout="wide")

st.markdown("# Settings")

try:
    from services.database import (
        get_supabase_client, list_configs, upsert_config, get_default_config,
        set_default_config, update_config,
        list_resumes, get_db_connection, save_chunks, update_resume,
        delete_chunks_and_embeddings, save_extracted_fields,
    )
    from services.database import save_embeddings_batch
    from services.parser import chunk_text
    from services.embeddings import generate_embeddings_batch
    from services.ai_engine import extract_fields
    from services.utils import load_default_extraction_config

    client = get_supabase_client()

    tab_config, tab_reparse = st.tabs(["Extraction Config", "Batch Re-Parse"])

    # Extraction Config
    with tab_config:
        st.caption("Configure which fields are extracted from resumes during parsing. Only the fields listed here will be extracted.")

        all_configs = list_configs(client)
        db_config = get_default_config(client)

        if not db_config and not all_configs:
            default = load_default_extraction_config()
            current_fields = default.get("fields", [])
            config_name = "Default"
            config_id = None
        elif db_config:
            current_fields = db_config.get("fields", [])
            config_name = db_config.get("name", "Default")
            config_id = db_config.get("id")
        else:
            current_fields = all_configs[0].get("fields", [])
            config_name = all_configs[0].get("name", "Default")
            config_id = all_configs[0].get("id")

        # Config selector
        if all_configs and len(all_configs) > 1:
            st.markdown("#### Saved Configurations")
            config_options = {
                f"{c['name']}{'  [active]' if c.get('is_default') else ''}": c
                for c in all_configs
            }

            sel_col1, sel_col2 = st.columns([4, 1])
            with sel_col1:
                selected_config_label = st.selectbox(
                    "Select config to edit",
                    list(config_options.keys()),
                    label_visibility="collapsed",
                )
            selected_cfg = config_options[selected_config_label]

            with sel_col2:
                if not selected_cfg.get("is_default"):
                    if st.button("Set as Active"):
                        set_default_config(client, selected_cfg["id"])
                        st.rerun()
                else:
                    st.caption("Active")

            current_fields = selected_cfg.get("fields", [])
            config_name = selected_cfg.get("name", "Default")
            config_id = selected_cfg.get("id")

            st.markdown("---")

        st.caption(f"Editing: {config_name}")

        if "config_fields" not in st.session_state or st.session_state.get("_config_id") != config_id:
            st.session_state.config_fields = [f.copy() for f in current_fields]
            st.session_state._config_id = config_id

        fields = st.session_state.config_fields

        # Column headers
        hcol1, hcol2, hcol3, hcol4, hcol5 = st.columns([3, 2, 2, 1, 1])
        with hcol1:
            st.caption("Key")
        with hcol2:
            st.caption("Label")
        with hcol3:
            st.caption("Type")
        with hcol4:
            st.caption("Required")
        with hcol5:
            st.caption("")

        for i, field in enumerate(fields):
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])

            with col1:
                fields[i]["key"] = st.text_input(
                    "Key", value=field.get("key", ""), key=f"cf_key_{i}",
                    label_visibility="collapsed", placeholder="field_key"
                )
            with col2:
                fields[i]["label"] = st.text_input(
                    "Label", value=field.get("label", ""), key=f"cf_label_{i}",
                    label_visibility="collapsed", placeholder="Display Label"
                )
            with col3:
                fields[i]["type"] = st.selectbox(
                    "Type", ["text", "number", "list"],
                    index=["text", "number", "list"].index(field.get("type", "text")),
                    key=f"cf_type_{i}", label_visibility="collapsed"
                )
            with col4:
                fields[i]["required"] = st.checkbox(
                    "Req", value=field.get("required", False), key=f"cf_req_{i}"
                )
            with col5:
                if st.button("Remove", key=f"cf_del_{i}"):
                    fields.pop(i)
                    st.rerun()

        if st.button("Add Field"):
            fields.append({"key": "", "label": "", "type": "text", "required": False})
            st.rerun()

        st.markdown("---")

        save_col1, save_col2, save_col3 = st.columns([2, 1.5, 1.5])

        with save_col1:
            new_config_name = st.text_input("Config Name", value=config_name)

        with save_col2:
            st.markdown("")
            st.markdown("")
            if st.button("Save", type="primary"):
                valid_fields = [f for f in fields if f.get("key") and f.get("label")]
                if not valid_fields:
                    st.error("Add at least one field with a key and label.")
                else:
                    try:
                        if config_id:
                            update_config(client, config_id, new_config_name, valid_fields)
                        else:
                            upsert_config(client, new_config_name, valid_fields, is_default=True)
                        st.success("Configuration saved. Re-parse resumes to apply changes.")
                        st.session_state.config_fields = valid_fields
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save: {str(e)}")

        with save_col3:
            st.markdown("")
            st.markdown("")
            if st.button("Save as New"):
                valid_fields = [f for f in fields if f.get("key") and f.get("label")]
                if not valid_fields:
                    st.error("Add at least one field with a key and label.")
                elif not new_config_name or new_config_name == config_name:
                    st.error("Enter a different name for the new config.")
                else:
                    try:
                        upsert_config(client, new_config_name, valid_fields, is_default=True)
                        st.success(f'New config "{new_config_name}" created and set as active.')
                        st.session_state._config_id = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create: {str(e)}")

        with st.expander("JSON Preview"):
            valid_fields = [f for f in fields if f.get("key")]
            st.json({"fields": valid_fields})

    # Batch Re-Parse
    with tab_reparse:
        st.caption("Re-chunk, re-embed, and re-extract fields for all parsed resumes using the active extraction config.")

        resumes = list_resumes(client)
        parsed_resumes = [r for r in resumes if r.get("status") == "parsed" and r.get("raw_text")]

        st.caption(f"{len(parsed_resumes)} resumes available for re-processing.")

        if parsed_resumes and st.button("Re-Parse All Resumes", type="primary"):
            progress = st.progress(0, text="Starting...")

            db_config = get_default_config(client)
            config_fields = db_config.get("fields", []) if db_config else load_default_extraction_config().get("fields", [])
            config_id_reparse = db_config["id"] if db_config else None

            conn = get_db_connection()
            success_count = 0

            try:
                for idx, resume in enumerate(parsed_resumes):
                    name = resume.get("candidate_name") or resume.get("filename", "")
                    progress.progress(
                        (idx + 1) / len(parsed_resumes),
                        text=f"Re-parsing {name} ({idx+1}/{len(parsed_resumes)})"
                    )

                    try:
                        raw_text = resume["raw_text"]
                        resume_id = resume["id"]

                        delete_chunks_and_embeddings(client, conn, resume_id)

                        chunks = chunk_text(raw_text)
                        saved_chunks = save_chunks(client, resume_id, chunks)

                        chunk_texts = [c["chunk_text"] for c in chunks]
                        embeddings = generate_embeddings_batch(chunk_texts)

                        embedding_rows = [
                            (saved_chunks[i]["id"], resume_id, embeddings[i])
                            for i in range(len(saved_chunks))
                        ]
                        save_embeddings_batch(conn, embedding_rows)

                        extracted = extract_fields(raw_text, config_fields)
                        if config_id_reparse:
                            save_extracted_fields(client, resume_id, config_id_reparse, extracted)

                        # Compute and save text_hash for cross-format dedup
                        from services.duplicate import compute_text_hash
                        text_hash = compute_text_hash(raw_text)

                        update_data = {"text_hash": text_hash}
                        if extracted.get("candidate_name"):
                            update_data["candidate_name"] = extracted["candidate_name"]
                        if extracted.get("email"):
                            update_data["candidate_email"] = extracted["email"]
                        if update_data:
                            update_resume(client, resume_id, update_data)

                        success_count += 1

                    except Exception as e:
                        st.warning(f"Failed: {name}: {str(e)[:80]}")

            finally:
                conn.close()

            progress.progress(1.0, text="Re-parse complete")
            st.success(f"Re-parsed {success_count}/{len(parsed_resumes)} resumes.")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.caption("Check your .env configuration.")
