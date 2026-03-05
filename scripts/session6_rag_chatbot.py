#!/usr/bin/env python3
"""
Session 6: Production RAG Chatbot CLI
Template style aligned with scripts/session4_chatbot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm import RuntimeLLMConfig
from utils.messages import ChatMessage
from utils.logger import get_logger

from src.rag.production_service import (
    ProductionRAGService,
    RAGBuildConfig,
    RAGQueryConfig,
)

log = get_logger(__name__)


def print_header() -> None:
    print("\n" + "=" * 76)
    print("📚 SESSION 6 - PRODUCTION RAG CHATBOT")
    print("=" * 76)
    print("\nIndexes all PDFs in data/ and saves vector store to data/rag_pdf_store.pkl")
    print("\nCommands:")
    print("  - 'build' : Build / rebuild vector store from all data/*.pdf")
    print("  - 'load'  : Load existing vector store")
    print("  - 'stats' : Show current session/vector store stats")
    print("  - 'reset' : Clear chat history")
    print("  - 'quit' or 'exit' : End conversation")
    print()


def main() -> None:
    print_header()

    service = ProductionRAGService()
    history: list[ChatMessage] = []
    store = None

    build_config = RAGBuildConfig(chunk_method="size", chunk_size=900, overlap=150)
    query_config = RAGQueryConfig(top_k=5, context_char_budget=3200, keep_last_n=6)

    runtime = RuntimeLLMConfig(temperature=0.2, max_new_tokens=256, top_p=0.95, seed=42)

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in {"quit", "exit"}:
                print("\n👋 Goodbye!\n")
                break

            if cmd == "reset":
                history = []
                print("\n🔄 Conversation history cleared\n")
                continue

            if cmd == "build":
                res = service.build_store_from_data_pdfs(build_config)
                store = service.load_store()
                print(
                    f"\n✅ Built {res.store_path.name} | documents={res.total_documents} | chunks={res.total_chunks}\n"
                )
                continue

            if cmd == "load":
                store = service.load_store()
                print(f"\n✅ Loaded {service.store_path.name} with {len(store.chunks)} chunks\n")
                continue

            if cmd == "stats":
                turn_count = max(0, (len([m for m in history if m.role != "system"]) // 2))
                store_chunks = len(store.chunks) if store is not None else 0
                print("\n📊 Stats:")
                print(f"   Turns: {turn_count}")
                print(f"   Messages: {len(history)}")
                print(f"   Loaded chunks: {store_chunks}")
                print(
                    f"   Query config: top_k={query_config.top_k}, budget={query_config.context_char_budget}, keep_last_n={query_config.keep_last_n}"
                )
                print(
                    f"   Runtime: temp={runtime.temperature}, max_new_tokens={runtime.max_new_tokens}, top_p={runtime.top_p}, seed={runtime.seed}"
                )
                print()
                continue

            if store is None:
                print("\n⚠️ No vector store loaded. Run 'build' or 'load' first.\n")
                continue

            print("\n🤖 Assistant: ", end="", flush=True)
            result = service.answer(
                user_input,
                history,
                store,
                runtime=runtime,
                query_config=query_config,
            )
            history = result.history
            print(result.answer + "\n")

            print("🔎 Retrieval trace:")
            for i, item in enumerate(result.trace.retrieved, 1):
                src = item.metadata.get("source", "unknown")
                print(
                    f"   {i}. score={item.score:.4f} | vector_id={item.vector_id} | chunk_id={item.chunk_id} | source={src}"
                )
            print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except ValueError as e:
            print(f"\n⚠️ {e}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            log.error("RAG chatbot error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
