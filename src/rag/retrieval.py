from __future__ import annotations
import numpy as np

def top_k(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int = 4) -> list[int]:
    sims = doc_vecs @ query_vec  # cosine with normalized vectors
    return np.argsort(-sims)[:k].tolist()

def build_context(chunks: list[str], idxs: list[int], char_budget: int = 3200) -> tuple[str, list[str]]:
    selected: list[str] = []
    total = 0
    for i in idxs:
        c = chunks[i]
        if total + len(c) > char_budget:
            break
        selected.append(c)
        total += len(c)
    return "\n\n---\n\n".join(selected), selected
