"""
Session 4-style Production RAG Chatbot Web UI

Features:
- Ingest all PDFs from data/ into a vector store in data/
- Configurable chunking: size-based with overlap OR semantic
- Cosine-similarity retrieval with transparent stage-level trace
- Runtime LLM controls: temperature, max tokens, top_p, seed
"""

from __future__ import annotations

import streamlit as st

from utils.llm import RuntimeLLMConfig
from utils.messages import ChatMessage
from src.rag.production_service import (
    ProductionRAGService,
    RAGBuildConfig,
    RAGQueryConfig,
)

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 Session 4 - Production RAG Chatbot")
st.caption("All PDFs in data/ → vector store in data/rag_pdf_store.pkl")

service = ProductionRAGService()

if "rag_history_prod" not in st.session_state:
    st.session_state.rag_history_prod = []
if "rag_store_prod" not in st.session_state:
    st.session_state.rag_store_prod = None
if "rag_last_trace" not in st.session_state:
    st.session_state.rag_last_trace = None

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("LLM Runtime")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 1024, 256, 32)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05)
    seed_enabled = st.checkbox("Use deterministic seed", value=True)
    seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)

    st.divider()

    st.subheader("Chunking")
    chunk_method = st.selectbox("Chunking method", ["size", "semantic"], index=0)
    chunk_size = st.slider("Chunk size (chars)", 300, 2400, 900, 50)
    overlap = st.slider("Overlap (chars)", 0, 600, 150, 10)

    st.divider()

    st.subheader("Retrieval")
    top_k = st.slider("Top-k", 1, 15, 5, 1)
    char_budget = st.slider("Context budget (chars)", 500, 12000, 3200, 100)
    keep_last_n = st.slider("Conversation memory (turns)", 0, 20, 6, 1)

    st.divider()
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.rag_history_prod = []
        st.session_state.rag_last_trace = None
        st.rerun()

st.markdown("### 1) Data readiness")
pdfs = service.list_pdf_documents()
if not pdfs:
    st.warning("No PDF files found in data/. Add PDFs to data/ and click build.")
else:
    st.success(f"Found {len(pdfs)} PDF file(s) in data/.")
    with st.expander("View discovered PDFs"):
        for p in pdfs:
            st.markdown(f"- {p.name}")

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("🧱 Build / Rebuild Vector Store", use_container_width=True):
        try:
            build_cfg = RAGBuildConfig(
                chunk_method=chunk_method,
                chunk_size=int(chunk_size),
                overlap=int(overlap),
            )
            result = service.build_store_from_data_pdfs(build_cfg)
            st.session_state.rag_store_prod = service.load_store()
            st.success(
                f"Built store: {result.store_path.name} | documents={result.total_documents} | chunks={result.total_chunks}"
            )
        except Exception as e:
            st.error(f"Build failed: {e}")

with col_b:
    if st.button("📦 Load Existing Vector Store", use_container_width=True):
        try:
            st.session_state.rag_store_prod = service.load_store()
            st.success(f"Loaded: {service.store_path.name} ({len(st.session_state.rag_store_prod.chunks)} chunks)")
        except Exception as e:
            st.error(f"Load failed: {e}")

st.markdown("### 2) Chat")
store = st.session_state.rag_store_prod
tab_chat, tab_trace, tab_chunks = st.tabs(["💬 Chat", "🔎 RAG Trace", "🧩 Chunk Explorer"])

with tab_chat:
    st.markdown("### 2) Chat")
    if store is None:
        st.info("Build or load the vector store first.")
    else:
        history = st.session_state.rag_history_prod
        for msg in history:
            if msg.role == "system":
                continue
            with st.chat_message(msg.role):
                st.markdown(msg.content)

        prompt = st.chat_input("Ask about your PDF documents")
        if prompt:
            runtime = RuntimeLLMConfig(
                temperature=float(temperature),
                max_new_tokens=int(max_tokens),
                top_p=float(top_p),
                seed=int(seed) if seed_enabled else None,
            )
            query_cfg = RAGQueryConfig(
                top_k=int(top_k),
                context_char_budget=int(char_budget),
                keep_last_n=int(keep_last_n),
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating..."):
                    try:
                        result = service.answer(
                            prompt,
                            history,
                            store,
                            runtime=runtime,
                            query_config=query_cfg,
                        )
                        st.markdown(result.answer)
                        st.session_state.rag_history_prod = result.history
                        st.session_state.rag_last_trace = result.trace
                    except Exception as e:
                        st.error(f"RAG query failed: {e}")

with tab_trace:
    st.markdown("### 3) RAG Stages (trace)")
    trace = st.session_state.rag_last_trace
    if trace is None:
        st.caption("Run one query to view stage-level details.")
    else:
        with st.expander("Stage A - Query vectorisation", expanded=True):
            st.markdown(f"- Query: {trace.query}")
            st.markdown(f"- Embedding dimension: {trace.query_vector_dimension}")
            st.markdown(f"- L2 norm: {trace.query_vector_norm:.6f}")
            st.markdown(f"- Preview (first 12 dims): {trace.query_vector_preview}")

        with st.expander("Stage B - Chunk retrieval (cosine)", expanded=True):
            for rank, item in enumerate(trace.retrieved, 1):
                source = item.metadata.get("source", "unknown")
                st.markdown(
                    f"**#{rank}** | score={item.score:.4f} | vector_id={item.vector_id} | chunk_id={item.chunk_id} | source={source}"
                )
                with st.container(border=True):
                    st.markdown(item.text)

        with st.expander("Stage C - Context passed to LLM", expanded=False):
            st.text_area("Context", value=trace.context_text, height=260)

with tab_chunks:
    st.markdown("### 4) Chunk + Vector Store Viewer")
    if store is None:
        st.info("Build or load the vector store first.")
    else:
        total_chunks = len(store.chunks)
        vector_dim = int(store.vectors.shape[1]) if store.vectors.ndim == 2 and total_chunks > 0 else 0
        st.caption(f"Total chunks: {total_chunks} | Embedding dimension: {vector_dim}")

        rows = []
        for vector_id, (chunk_text, metadata) in enumerate(zip(store.chunks, store.metadatas)):
            rows.append(
                {
                    "vector_id": int(metadata.get("vector_id", vector_id)),
                    "chunk_id": str(metadata.get("chunk_id", f"chunk-{vector_id}")),
                    "source": metadata.get("source", "unknown"),
                    "chunk_index": int(metadata.get("chunk_index", -1)),
                    "char_len": len(chunk_text),
                    "text_preview": chunk_text[:180].replace("\n", " "),
                }
            )

        st.dataframe(rows, use_container_width=True, hide_index=True)

        selected_vector_id = st.selectbox(
            "Inspect vector by vector_id",
            options=list(range(total_chunks)),
            index=0,
            format_func=lambda idx: f"vector_id={idx} | chunk_id={store.metadatas[idx].get('chunk_id', f'chunk-{idx}')}",
        )

        selected_meta = store.metadatas[selected_vector_id]
        selected_chunk_id = str(selected_meta.get("chunk_id", f"chunk-{selected_vector_id}"))
        selected_source = selected_meta.get("source", "unknown")
        selected_vector = store.vectors[selected_vector_id].tolist()

        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"- vector_id: {selected_vector_id}")
            st.markdown(f"- chunk_id: {selected_chunk_id}")
            st.markdown(f"- source: {selected_source}")
            st.markdown(f"- vector_norm: {float((store.vectors[selected_vector_id] ** 2).sum() ** 0.5):.6f}")
        with right:
            st.markdown("**Chunk text**")
            st.text_area("Selected chunk", value=store.chunks[selected_vector_id], height=180)

        st.markdown("**Vector values**")
        show_full_vector = st.checkbox("Show full vector", value=False)
        if show_full_vector:
            st.json(selected_vector)
        else:
            st.json(selected_vector[:32])
