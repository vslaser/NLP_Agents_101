from __future__ import annotations
from functools import lru_cache
import numpy as np
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def _model():
    # Small + fast on CPU
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list[str]) -> np.ndarray:
    return np.array(_model().encode(texts, normalize_embeddings=True))

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
