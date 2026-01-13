# nlp_llm_agents

A session-by-session course repo that grows from:
- Python foundations → prompts → chatbot → RAG → agents → multi-agents (LangGraph)

## Quickstart

```bash
poetry install
cp .env.example .env
poetry run streamlit run ui/app.py
```

## Notes
- Local LLM is CPU-only via `transformers`.
- API mode uses an OpenAI-compatible endpoint (set `OPENAI_COMPAT_*` in `.env`).
