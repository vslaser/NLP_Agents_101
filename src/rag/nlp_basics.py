from __future__ import annotations
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

def bow_demo(texts: list[str]) -> tuple[list[str], np.ndarray]:
    vec = CountVectorizer()
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out().tolist()
    return vocab, X.toarray()

def tokenization_demo(model_id: str, text: str) -> dict:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    ids = tok.encode(text)
    tokens = tok.convert_ids_to_tokens(ids)
    return {"ids": ids, "tokens": tokens}

def embedding_demo(sentences: list[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return np.array(model.encode(sentences, normalize_embeddings=True))

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
