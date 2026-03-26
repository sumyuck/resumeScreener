"""
Utility helpers: hashing, text normalization, formatting.
"""

import hashlib
import re
import json
import os
from pathlib import Path


def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def safe_json(obj, default=None):
    if obj is None:
        return default
    if isinstance(obj, (dict, list)):
        return obj
    try:
        return json.loads(obj)
    except (json.JSONDecodeError, TypeError):
        return default


def load_default_extraction_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "default_extraction.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {"fields": []}


def score_color(score: float) -> str:
    if score >= 7:
        return "green"
    elif score >= 4:
        return "orange"
    return "red"


def confidence_label(confidence: str) -> str:
    return {"high": "High", "medium": "Medium", "low": "Low"}.get(confidence, "Unknown")


def truncate(text: str, max_len: int = 200) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(' ', 1)[0] + "..."


def extract_skills_from_text(text: str) -> list[str]:
    common_skills = [
        "python", "java", "javascript", "typescript", "react", "angular", "vue",
        "node.js", "nodejs", "express", "django", "flask", "fastapi",
        "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "machine learning", "deep learning", "nlp", "computer vision",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy",
        "git", "ci/cd", "jenkins", "github actions",
        "rest", "graphql", "grpc", "microservices",
        "html", "css", "tailwind", "sass",
        "c++", "c#", "go", "rust", "kotlin", "swift",
        "figma", "jira", "confluence", "agile", "scrum",
        "linux", "bash", "shell scripting",
        "spark", "hadoop", "airflow", "kafka",
        "streamlit", "tableau", "power bi",
        "supabase", "firebase", "heroku", "vercel",
        "langchain", "gemini", "faiss", "bm25", "opencv", "spring boot",
        "convlstm", "unet", "groq", "pinecone", "chromadb",
        "postman", "render", "rag", "llm",
    ]
    text_lower = text.lower()
    found = []
    for skill in common_skills:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return found
