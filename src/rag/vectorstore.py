from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np

from src.rag.chunking import Chunk


@dataclass
class PickleVectorStore:
    """A tiny, file-backed 'vector store' for the course.

    Stored on disk as a pickle containing:
    - chunks: list[str]
    - vectors: np.ndarray (float32)
    - metadatas: list[dict]
    """

    chunks: list[str]
    vectors: np.ndarray
    metadatas: list[dict]

    def __post_init__(self):
        if len(self.chunks) != len(self.metadatas) or len(self.chunks) != len(self.vectors):
            raise ValueError("VectorStore length mismatch (chunks / vectors / metadatas).")

    @staticmethod
    def from_chunks(chunks: list[Chunk], vectors: np.ndarray) -> "PickleVectorStore":
        return PickleVectorStore(
            chunks=[c.text for c in chunks],
            vectors=vectors.astype(np.float32),
            metadatas=[c.meta for c in chunks],
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "vectors": self.vectors,
                    "metadatas": self.metadatas,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return path

    @staticmethod
    def load(path: str | Path) -> "PickleVectorStore":
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return PickleVectorStore(
            chunks=list(obj["chunks"]),
            vectors=np.array(obj["vectors"], dtype=np.float32),
            metadatas=list(obj["metadatas"]),
        )
