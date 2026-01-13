from __future__ import annotations

import streamlit as st
import numpy as np
from langchain_core.messages import AIMessage, ToolMessage

from utils.settings import load_settings
from utils.llm import RuntimeLLMConfig
from utils.prompt_loader import load_prompt
from utils.prompting import render
from utils.messages import ChatMessage

from src.chatbot.service import answer as chatbot_answer
from src.rag.service import ingest_text, ingest_pdf, load_store, list_vectorstores, answer_with_rag
from src.agents.single_agent import run_single_agent
from src.agents.multi_agent import run_multi_agent

st.set_page_config(page_title="nlp_llm_agents", page_icon="🧠", layout="wide")
st.title("nlp_llm_agents")

settings = load_settings()

# Runtime knobs (Session 4.5) - wired into Local LLM calls for chatbot + RAG
with st.sidebar:
    st.header("Runtime Controls")
    st.write(f"Mode: **{settings.llm_mode}**")
    temp = st.slider("Temperature", 0.0, 1.0, float(settings.local_temperature), 0.05)
    max_new = st.slider("Max new tokens", 32, 512, int(settings.local_max_new_tokens), 16)
    st.caption("For API mode, temperature is fixed to 0 in the agent tool-calling model.")
    runtime = RuntimeLLMConfig(temperature=temp, max_new_tokens=max_new)

tab_chatbot, tab_rag, tab_agents, tab_prompts = st.tabs(["Chatbot", "RAG", "Agents", "Prompt Playground"])

# ---------------- Chatbot tab (Session 4) ----------------
with tab_chatbot:
    st.subheader("Chatbot (Local CPU or API)")

    # Separate short-term memory control for plain chatbot
    chat_keep_last = st.slider("Chatbot short-term memory (turns to keep)", 0, 30, 10, 1, key="chat_keep_last")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for m in st.session_state.chat_history:
        if m.role == "system":
            continue
        with st.chat_message(m.role):
            st.markdown(m.content)

    prompt = st.chat_input("Ask something", key="chat_input")
    if prompt:
        out, hist = chatbot_answer(prompt, st.session_state.chat_history, runtime=runtime, keep_last_n=chat_keep_last)
        st.session_state.chat_history = hist
        st.rerun()


with tab_rag:
    st.subheader("RAG Chatbot (PDF/TXT → Pickle Vector Store)")

    if "rag_store_path" not in st.session_state:
        st.session_state.rag_store_path = None
    if "rag_history" not in st.session_state:
        st.session_state.rag_history = []
    if "rag_store" not in st.session_state:
        st.session_state.rag_store = None

    st.markdown("### 1) Ingest documents (build a pickle vector store)")
    colA, colB, colC = st.columns([1.2, 1, 1])

    with colA:
        uploaded = st.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"], key="rag_upload")
        store_name = st.text_input("Store name (no spaces)", value="my_docs", key="rag_store_name")
    with colB:
        chunk_method = st.selectbox("Chunking method", ["size", "semantic"], index=0, key="rag_chunk_method")
        chunk_size = st.slider("Chunk size (chars)", 300, 2000, 900, 50, key="rag_chunk_size")
        overlap = st.slider("Overlap (chars)", 0, 400, 150, 10, key="rag_overlap")
    with colC:
        st.caption("Tip: start with **size** chunking. Switch to **semantic** when documents have clear paragraphs/headings.")
        ingest_clicked = st.button("Ingest → Save .pkl", key="rag_ingest_btn", use_container_width=True)

    if ingest_clicked:
        if not uploaded:
            st.warning("Upload a PDF or TXT first.")
        else:
            safe_name = store_name.strip().replace(" ", "_")
            if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
                res = ingest_pdf(
                    uploaded.read(),
                    store_name=safe_name,
                    chunk_method=chunk_method,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            else:
                text = uploaded.read().decode("utf-8", errors="ignore")
                res = ingest_text(
                    text,
                    store_name=safe_name,
                    chunk_method=chunk_method,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            st.session_state.rag_store_path = str(res.store_path)
            st.session_state.rag_store = res.store
            st.success(f"Saved vector store: {res.store_path.name} ({res.total_chunks} chunks)")

    st.divider()
    st.markdown("### 2) Load an existing vector store")
    stores = list_vectorstores()
    store_labels = ["(none)"] + [p.name for p in stores]
    selected = st.selectbox("Pick a .pkl store", store_labels, index=0, key="rag_store_picker")

    if selected != "(none)":
        path = next(p for p in stores if p.name == selected)
        if st.button("Load selected store", key="rag_load_btn"):
            st.session_state.rag_store_path = str(path)
            st.session_state.rag_store = load_store(path)
            st.success(f"Loaded: {path.name} with {len(st.session_state.rag_store.chunks)} chunks")

    st.divider()
    st.markdown("### 3) Chat with RAG (grounded answers)")

    # short-term memory controls specific to RAG
    rag_keep_last = st.slider("RAG short-term memory (turns to keep)", 0, 20, 6, 1, key="rag_keep_last")
    topk = st.slider("Retrieve top-k", 1, 12, 5, 1, key="rag_topk")
    budget = st.slider("Context budget (chars)", 500, 8000, 3200, 100, key="rag_budget")

    if st.session_state.rag_store is None:
        st.info("Load or ingest a vector store above to start chatting.")
    else:
        for m in st.session_state.rag_history:
            if m.role == "system":
                continue
            with st.chat_message(m.role):
                st.markdown(m.content)

        prompt = st.chat_input("Ask about your documents", key="rag_input")
        if prompt:
            out, hist, used = answer_with_rag(
                prompt,
                st.session_state.rag_history,
                st.session_state.rag_store,
                runtime=runtime,
                keep_last_n=rag_keep_last,
                topk=topk,
                context_char_budget=budget,
            )
            st.session_state.rag_history = hist

            with st.expander("Retrieved context (debug)"):
                for i, (chunk, meta) in enumerate(used, 1):
                    st.markdown(f"**Chunk {i}** — {meta}")
                    st.markdown(chunk)

            st.rerun()


with tab_agents:
    st.subheader("Agents (Single-agent and Multi-agent)")

    mode = st.selectbox("Agent mode", ["Single Agent", "Multi-Agent (Orchestrator)"], key="agent_mode")
    show_trace = st.checkbox("Show execution trace (debug)", value=True, key="agent_trace")

    # Separate short-term memory controls for agents
    if mode == "Single Agent":
        keep_last = st.slider("Single-agent short-term memory (messages to keep)", 0, 30, 10, 1, key="single_keep_last")
    else:
        keep_last = st.slider("Multi-agent short-term memory (messages to keep)", 0, 40, 12, 1, key="multi_keep_last")

    # Maintain separate histories
    if "single_agent_history" not in st.session_state:
        st.session_state.single_agent_history = []
    if "multi_agent_history" not in st.session_state:
        st.session_state.multi_agent_history = []

    def _render_lc_message(msg):
        # Map LangChain message types to Streamlit chat roles
        if isinstance(msg, ToolMessage):
            return ("assistant", f"**[tool:{msg.name}]**\n\n{msg.content}")
        if isinstance(msg, AIMessage):
            return ("assistant", msg.content or "")
        # HumanMessage / others
        role = "user"
        content = getattr(msg, "content", "")
        return (role, content)

    history = st.session_state.single_agent_history if mode == "Single Agent" else st.session_state.multi_agent_history

    st.markdown("### Conversation")
    for msg in history:
        role, content = _render_lc_message(msg)
        with st.chat_message(role):
            st.markdown(content)

    user_in = st.chat_input("Ask the agent to do something (tools available: calculator, http_get)", key="agent_input")
    if user_in:
        if mode == "Single Agent":
            out = run_single_agent(user_in, history=history, keep_last_n=keep_last)
            new_hist = list(out["messages"])
            st.session_state.single_agent_history = new_hist
        else:
            out = run_multi_agent(user_in, history=history, keep_last_n=keep_last)
            new_hist = list(out["messages"])
            st.session_state.multi_agent_history = new_hist

        if show_trace:
            with st.expander("Execution trace (raw messages)"):
                for i, msg in enumerate(new_hist, 1):
                    st.write(i, type(msg).__name__, getattr(msg, "content", ""))

        st.rerun()


with tab_prompts:
    st.subheader("Prompt Playground (compile prompts without calling the model)")

    task = st.text_area("Task", "Summarize the text.")
    context = st.text_area("Context", "Large language models are probabilistic text generators...")
    fmt = st.text_input("Format instructions", "Return 3 bullet points.")

    template = load_prompt("prompt_playground_task.txt")
    compiled = render(template, task=task, context=context, format_instructions=fmt)

    st.markdown("### Compiled Prompt")
    st.code(compiled)
