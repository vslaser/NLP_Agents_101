from __future__ import annotations

from pathlib import Path
import numpy as np

from utils.llm import get_llm, RuntimeLLMConfig
from utils.messages import ChatMessage
from utils.prompt_loader import load_prompt
from utils.memory_profiles import rag_short_term

from src.rag.embeddings import embed_query
from src.rag.retrieval import top_k, build_context
from src.rag.vectorstore import PickleVectorStore
from src.rag.ingestion import ingest_text_to_pickle, ingest_pdf_to_pickle, IngestionResult


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "vectorstores"


def _render_rag_prompt(system_prompt: str, context: str) -> str:
    template = load_prompt("rag_context_template.txt")
    return (
        template
        .replace("[[SYSTEM_PROMPT]]", system_prompt)
        .replace("[[CONTEXT]]", context)
    )


def list_vectorstores() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.pkl"))


def load_store(path: str | Path) -> PickleVectorStore:
    return PickleVectorStore.load(path)


def ingest_text(
    text: str,
    *,
    store_name: str,
    chunk_method: str = "size",
    chunk_size: int = 900,
    overlap: int = 150,
) -> IngestionResult:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store_path = DATA_DIR / f"{store_name}.pkl"
    return ingest_text_to_pickle(
        raw_text=text,
        store_path=store_path,
        source_name=store_name,
        chunk_method=chunk_method,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def ingest_pdf(
    pdf_bytes: bytes,
    *,
    store_name: str,
    chunk_method: str = "size",
    chunk_size: int = 900,
    overlap: int = 150,
) -> IngestionResult:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store_path = DATA_DIR / f"{store_name}.pkl"
    return ingest_pdf_to_pickle(
        pdf_bytes=pdf_bytes,
        store_path=store_path,
        source_name=store_name,
        chunk_method=chunk_method,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def answer_with_rag(
    user_text: str,
    history: list[ChatMessage],
    store: PickleVectorStore,
    runtime: RuntimeLLMConfig | None = None,
    *,
    keep_last_n: int = 6,
    topk: int = 5,
    context_char_budget: int = 3200,
) -> tuple[str, list[ChatMessage], list[tuple[str, dict]]]:
    """Return (answer, updated_history, retrieved_chunks_with_meta)."""
    system = load_prompt("rag_system.txt")
    if not history:
        history = [ChatMessage(role="system", content=system)]
    history.append(ChatMessage(role="user", content=user_text))

    qv = embed_query(user_text)
    idxs = top_k(qv, store.vectors, k=topk)
    context, used_chunks = build_context(store.chunks, idxs, char_budget=context_char_budget)
    used = [(store.chunks[i], store.metadatas[i]) for i in idxs[: len(used_chunks)]]

    rag_prompt = _render_rag_prompt(system, context)
    msgs = [ChatMessage(role="system", content=rag_prompt)]
    msgs += [m for m in rag_short_term(history, keep_last_n=keep_last_n) if m.role != "system"]

    llm = get_llm(runtime)
    out = llm.chat(msgs)
    history.append(ChatMessage(role="assistant", content=out))
    return out, history, used
