"""
Embeddings service: HuggingFace Inference API for text embeddings.
Uses sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"
EMBEDDING_DIM = 384

_cache: dict[str, list[float]] = {}


def _get_headers() -> dict:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN must be set in .env")
    return {"Authorization": f"Bearer {token}"}


def _embed_texts(texts: list[str]) -> list[list[float]]:
    truncated = [t[:8000] if len(t) > 8000 else t for t in texts]

    uncached_indices = []
    uncached_texts = []
    results = [None] * len(truncated)

    for i, text in enumerate(truncated):
        key = text[:500]
        if key in _cache:
            results[i] = _cache[key]
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        try:
            response = requests.post(
                HF_API_URL,
                headers=_get_headers(),
                json={"inputs": uncached_texts, "options": {"wait_for_model": True}},
                timeout=60,
            )
            response.raise_for_status()
            embeddings = response.json()

            for idx, emb in zip(uncached_indices, embeddings):
                _cache[truncated[idx][:500]] = emb
                results[idx] = emb
        except Exception:
            for idx in uncached_indices:
                if results[idx] is None:
                    results[idx] = [0.0] * EMBEDDING_DIM

    return results


def generate_embedding(text: str) -> list[float]:
    return _embed_texts([text])[0]


def generate_query_embedding(text: str) -> list[float]:
    return generate_embedding(text)


def generate_embeddings_batch(texts: list[str], task_type: str = None) -> list[list[float]]:
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        all_embeddings.extend(_embed_texts(batch))
    return all_embeddings
