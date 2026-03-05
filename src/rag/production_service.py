from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np

from utils.llm import RuntimeLLMConfig, get_llm
from utils.messages import ChatMessage
from utils.prompt_loader import load_prompt
from utils.memory_profiles import rag_short_term
from utils.exceptions import ExternalServiceError
from utils.logger import get_logger

from src.rag.chunking import chunk_text, Chunk
from src.rag.embeddings import embed_query, embed_texts
from src.rag.ingestion import extract_text_from_pdf
from src.rag.vectorstore import PickleVectorStore

log = get_logger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_STORE_PATH = DEFAULT_DATA_DIR / "rag_pdf_store.pkl"


def _render_grounded_rag_prompt(system_prompt: str, context: str) -> str:
    template = load_prompt("rag_grounded_context_template.txt")
    return (
        template
        .replace("[[SYSTEM_PROMPT]]", system_prompt)
        .replace("[[CONTEXT]]", context)
    )


@dataclass(frozen=True)
class RAGBuildConfig:
    chunk_method: str = "size"
    chunk_size: int = 900
    overlap: int = 150


@dataclass(frozen=True)
class RAGQueryConfig:
    top_k: int = 5
    min_similarity_score: float = 0.25
    context_char_budget: int = 3200
    keep_last_n: int = 6


@dataclass(frozen=True)
class RAGBuildResult:
    store_path: Path
    total_documents: int
    total_chunks: int


@dataclass(frozen=True)
class RetrievedChunk:
    vector_id: int
    score: float
    chunk_id: str
    text: str
    metadata: dict


@dataclass(frozen=True)
class RAGTrace:
    query: str
    query_vector_dimension: int
    query_vector_norm: float
    query_vector_preview: list[float]
    retrieved: list[RetrievedChunk]
    context_text: str


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    history: list[ChatMessage]
    trace: RAGTrace


class ProductionRAGService:
    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        store_path: str | Path | None = None,
        system_prompt_name: str = "rag_system.txt",
    ):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.store_path = Path(store_path) if store_path else DEFAULT_STORE_PATH
        self.system_prompt_name = system_prompt_name

    def list_pdf_documents(self) -> list[Path]:
        if not self.data_dir.exists():
            return []
        return sorted(p for p in self.data_dir.glob("*.pdf") if p.is_file())

    def build_store_from_data_pdfs(self, config: RAGBuildConfig) -> RAGBuildResult:
        pdfs = self.list_pdf_documents()
        if not pdfs:
            raise ValueError(f"No PDF files found in {self.data_dir}")

        chunks: list[Chunk] = []
        doc_counter = 0

        for pdf_path in pdfs:
            raw = pdf_path.read_bytes()
            text, _ = extract_text_from_pdf(raw)
            text = text.strip()
            if not text:
                log.warning("Skipping empty PDF: %s", pdf_path.name)
                continue

            doc_counter += 1
            doc_id = self._document_id(pdf_path)
            base_meta = {
                "source": pdf_path.name,
                "source_path": str(pdf_path),
                "doc_id": doc_id,
                "chunking_method": config.chunk_method,
                "chunk_size": config.chunk_size,
                "overlap": config.overlap,
            }

            doc_chunks = chunk_text(
                text,
                method=config.chunk_method,
                chunk_size=config.chunk_size,
                overlap=config.overlap,
                base_meta=base_meta,
            )

            for index, chunk in enumerate(doc_chunks):
                chunk.meta["chunk_index"] = index
                chunk.meta["chunk_id"] = f"{doc_id}:{index:05d}"

            chunks.extend(doc_chunks)

        if not chunks:
            raise ValueError("No chunks were produced from the PDF documents.")

        vectors = embed_texts([c.text for c in chunks]).astype(np.float32)
        vectors = self._normalize_matrix(vectors)

        for vector_id, chunk in enumerate(chunks):
            chunk.meta["vector_id"] = vector_id

        store = PickleVectorStore.from_chunks(chunks, vectors)
        saved = store.save(self.store_path)
        return RAGBuildResult(store_path=saved, total_documents=doc_counter, total_chunks=len(chunks))

    def load_store(self) -> PickleVectorStore:
        if not self.store_path.exists():
            raise ValueError(f"Vector store not found at {self.store_path}")
        store = PickleVectorStore.load(self.store_path)
        store.vectors = self._normalize_matrix(store.vectors.astype(np.float32))
        return store

    def answer(
        self,
        user_text: str,
        history: list[ChatMessage],
        store: PickleVectorStore,
        runtime: RuntimeLLMConfig | None = None,
        query_config: RAGQueryConfig | None = None,
    ) -> RAGResponse:
        query_cfg = query_config or RAGQueryConfig()
        self._validate_input(user_text)

        system_prompt = load_prompt(self.system_prompt_name)
        if not history:
            history = [ChatMessage(role="system", content=system_prompt)]
        history.append(ChatMessage(role="user", content=user_text))

        query_vec = embed_query(user_text).astype(np.float32)
        query_vec = self._normalize_vector(query_vec)

        ranked = self._cosine_rank(query_vec, store.vectors, k=query_cfg.top_k)
        ranked = [
            (vector_id, score)
            for vector_id, score in ranked
            if score >= query_cfg.min_similarity_score
        ]
        retrieved = self._build_retrieved_chunks(ranked, store)
        context = self._build_context(retrieved, char_budget=query_cfg.context_char_budget)

        if not context.strip():
            answer = "I don't know."
            history.append(ChatMessage(role="assistant", content=answer))
            trace = RAGTrace(
                query=user_text,
                query_vector_dimension=int(query_vec.shape[0]),
                query_vector_norm=float(np.linalg.norm(query_vec)),
                query_vector_preview=[float(x) for x in query_vec[:12]],
                retrieved=retrieved,
                context_text=context,
            )
            return RAGResponse(answer=answer, history=history, trace=trace)

        rag_prompt = _render_grounded_rag_prompt(system_prompt, context)

        msgs = [ChatMessage(role="system", content=rag_prompt)]
        msgs += [m for m in rag_short_term(history, keep_last_n=query_cfg.keep_last_n) if m.role != "system"]

        try:
            llm = get_llm(runtime)
            answer = llm.chat(msgs)
        except Exception as e:
            log.error("RAG LLM call failed: %s", e, exc_info=True)
            raise ExternalServiceError(f"Failed to generate RAG response: {e}")

        if not answer or not answer.strip():
            raise ValueError("Empty model response")

        history.append(ChatMessage(role="assistant", content=answer))

        trace = RAGTrace(
            query=user_text,
            query_vector_dimension=int(query_vec.shape[0]),
            query_vector_norm=float(np.linalg.norm(query_vec)),
            query_vector_preview=[float(x) for x in query_vec[:12]],
            retrieved=retrieved,
            context_text=context,
        )
        return RAGResponse(answer=answer, history=history, trace=trace)

    def _build_context(self, retrieved: list[RetrievedChunk], *, char_budget: int) -> str:
        sections: list[str] = []
        total = 0
        for item in retrieved:
            header = (
                f"[chunk_id={item.chunk_id} | source={item.metadata.get('source')}"
                f" | score={item.score:.4f}]"
            )
            section = f"{header}\n{item.text}".strip()
            if total + len(section) > char_budget:
                break
            sections.append(section)
            total += len(section)
        return "\n\n---\n\n".join(sections)

    def _build_retrieved_chunks(
        self,
        ranked: list[tuple[int, float]],
        store: PickleVectorStore,
    ) -> list[RetrievedChunk]:
        out: list[RetrievedChunk] = []
        for vector_id, score in ranked:
            meta = store.metadatas[vector_id]
            out.append(
                RetrievedChunk(
                    vector_id=vector_id,
                    score=float(score),
                    chunk_id=str(meta.get("chunk_id", f"chunk-{vector_id}")),
                    text=store.chunks[vector_id],
                    metadata=meta,
                )
            )
        return out

    def _cosine_rank(self, query_vec: np.ndarray, vectors: np.ndarray, *, k: int) -> list[tuple[int, float]]:
        if vectors.ndim != 2:
            raise ValueError("Vector matrix must be 2-dimensional")
        if vectors.shape[0] == 0:
            return []
        if vectors.shape[1] != query_vec.shape[0]:
            raise ValueError("Embedding dimension mismatch between query and vector store")

        sims = vectors @ query_vec
        n = min(max(1, k), len(sims))
        idxs = np.argsort(-sims)[:n]
        return [(int(i), float(sims[i])) for i in idxs]

    def _validate_input(self, text: str) -> None:
        if not text.strip():
            raise ValueError("Empty input")
        if len(text) > 4000:
            raise ValueError("Input too long (max 4000 chars)")

    def _document_id(self, path: Path) -> str:
        digest = hashlib.sha1(path.name.encode("utf-8")).hexdigest()[:10]
        return f"doc-{digest}"

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return vector
        return vector / norm

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
