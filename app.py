"""
HR Resume Screener: Main Streamlit app.
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="HR Resume Screener",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_role_id = None
    st.session_state.current_resume_id = None
    st.session_state.user = None

# Sidebar
st.sidebar.markdown("## HR Resume Screener")
st.sidebar.markdown("---")

# Connection check
env_ok = all([
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY"),
    os.getenv("SUPABASE_DB_URL"),
    os.getenv("GROQ_API_KEY"),
    os.getenv("HF_TOKEN"),
])

if env_ok:
    st.sidebar.caption("All services connected")
else:
    missing = []
    for var in ["SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_DB_URL", "GROQ_API_KEY", "HF_TOKEN"]:
        if not os.getenv(var):
            missing.append(var)
    st.sidebar.error(f"Missing: {', '.join(missing)}")

# Default landing: redirect to Dashboard
st.switch_page("pages/1_Dashboard.py")
