#!/usr/bin/env python3
"""
Session 4: Production Chatbot CLI
Demonstrates evolution from Session 3 (pdf_chatbot.py) to production architecture

Key Concepts:
- Session 1: Clean project structure with modules
- Session 3: Prompt engineering and versioning
- Session 3.5: Explicit state, boundaries, failure isolation
- Session 4: Production-ready chatbot interface
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chatbot.service import ChatbotService
from src.chatbot.prompts import get_chatbot_v2_prompt
from utils.messages import ChatMessage
from utils.logger import get_logger
from utils.llm import RuntimeLLMConfig

log = get_logger(__name__)


def print_header():
    """Display chatbot header"""
    print("\n" + "="*70)
    print("💬 LLM CHATBOT - Session 4")
    print("="*70)
    print("\nDemonstrates: Service architecture, state management, error handling")
    print("\nCommands:")
    print("  - 'quit' or 'exit': End conversation")
    print("  - 'reset': Clear conversation history")
    print("  - 'stats': Show conversation statistics")
    print()


def handle_command(command: str, history: list[ChatMessage]) -> tuple[bool, list[ChatMessage]]:
    """
    Session 3.5: Handle commands at system boundary (not in LLM)
    
    Returns:
        (should_exit, updated_history)
    """
    cmd = command.lower()
    
    if cmd in ['quit', 'exit']:
        print("\n👋 Goodbye!\n")
        return True, history
    
    if cmd == 'reset':
        print("\n🔄 Conversation history cleared\n")
        return False, []
    
    if cmd == 'stats':
        turn_count = (len(history) - 1) // 2  # Exclude system message, count pairs
        print(f"\n📊 Stats:")
        print(f"   Conversation turns: {turn_count}")
        print(f"   Total messages: {len(history)}")
        print()
        return False, history
    
    return False, history


def main():
    """Main chatbot loop"""
    print_header()
    
    # Session 3.5: Explicit configuration
    system_prompt = get_chatbot_v2_prompt()
    
    # Initialize service with configuration
    service = ChatbotService(
        system_prompt=system_prompt,
        max_history_turns=10,
        max_input_length=2000
    )
    
    # Session 3.5: Explicit state
    history: list[ChatMessage] = []
    
    # Optional: Configure runtime parameters (temperature, max_tokens)
    runtime = RuntimeLLMConfig(temperature=0.7, max_new_tokens=256)
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Session 3.5: System boundary - handle commands in code
            if user_input.lower() in ['quit', 'exit', 'reset', 'stats']:
                should_exit, history = handle_command(user_input, history)
                if should_exit:
                    break
                continue
            
            # Process user message through service
            print("\n🤖 Assistant: ", end="", flush=True)
            response, history = service.chat(user_input, history, runtime)
            print(response + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
            
        except ValueError as e:
            # Session 3.5: Handle validation errors gracefully
            print(f"\n⚠️  {e}\n")
            
        except Exception as e:
            # Session 3.5: Failure isolation - errors don't crash the app
            print(f"\n❌ Error: {e}\n")
            log.error(f"Chat error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
