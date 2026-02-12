"""
Session 3.5 Refactor — Simple PDF Chatbot (Application Design Focus)

Goal of this refactor:
- Start from what students already saw in Session 3 (prompting with PDF content)
- Evolve the SAME file to demonstrate Session 3.5 concepts:
  1) Prompts live inside code (interfaces), not in loose strings
  2) State is explicit (chat is a loop with memory)
  3) System boundaries: code decides validation/routing; LLM does language
  4) Failure isolation: LLM calls are wrapped and errors don’t cascade
  5) Prompt versioning: prompts change over time; we make that explicit

This file intentionally keeps the PDF extraction and CLI shape familiar.
We refactor around it to make application structure visible and teachable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm import get_llm
from utils.settings import load_settings
from utils.messages import ChatMessage
from utils.logger import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------
# Prompt versioning (Session 3 concept, now made explicit in code)
# Why: prompts change, expectations drift, and debugging requires traceability.
# ------------------------------------------------------------
PROMPT_NAME = "pdf_chatbot"
PROMPT_VERSION = "v2.0"

# System prompt is an interface contract, not a casual instruction.
# We keep it as a constant to avoid “prompt drift” across the codebase.
SYSTEM_PROMPT = (
    "You are a helpful assistant. The user has provided a PDF document and will ask "
    "questions about it. Answer based on the document content provided. "
    "If the document does not contain the answer, say you cannot find it in the document."
)

# A smaller system prompt for follow-up turns once the document is already “known” in context.
FOLLOWUP_SYSTEM_PROMPT = (
    "You are a helpful assistant discussing a document with the user. "
    "Answer based on the document context that was provided earlier. "
    "If you cannot find the answer in the document, say so."
)

# Guardrails / boundaries (Session 3.5 concept)
MAX_USER_CHARS = 2000
DEFAULT_MAX_DOC_CHARS = 6000


# ------------------------------------------------------------
# Session 3.5: Explicit State
# Why: chatbots are loops with memory; state must be explicit, boring, and safe.
# ------------------------------------------------------------
@dataclass
class ConversationState:
    """
    Minimal, explicit chat state.
    Why a dataclass? It’s readable, structured, and easy for beginners.
    """
    history: List[ChatMessage]

    def __init__(self):
        self.history = []

    def clear(self) -> None:
        self.history = []

    def has_history(self) -> bool:
        return len(self.history) > 0

    def add(self, msg: ChatMessage) -> None:
        self.history.append(msg)


# ------------------------------------------------------------
# PDF extraction kept mostly as-is (not the focus of Session 3.5)
# ------------------------------------------------------------
def extract_pdf_text(pdf_path: str, max_chars: int = DEFAULT_MAX_DOC_CHARS) -> str:
    """Extract text from PDF file, truncated to fit model context."""
    try:
        import PyPDF2

        text_parts = []
        total_chars = 0

        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)

            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    page_text = f"--- Page {page_num} ---\n{text}"

                    # Check if adding this page would exceed limit
                    if total_chars + len(page_text) > max_chars:
                        remaining = max_chars - total_chars
                        if remaining > 200:  # Only add if there's meaningful space
                            text_parts.append(page_text[:remaining] + "\n\n[... truncated ...]")
                        break

                    text_parts.append(page_text)
                    total_chars += len(page_text)

            result = "\n\n".join(text_parts)

            # Print truncation info
            if total_chars >= max_chars or page_num < total_pages:
                print(
                    f"⚠️  Document truncated: Using first ~{len(result)} characters from "
                    f"{page_num}/{total_pages} pages"
                )
                print("   (Local model context limit: ~2000 tokens / ~8000 characters)\n")

            return result

    except ImportError:
        print("ERROR: PyPDF2 not installed. Run: pip install PyPDF2")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not read PDF: {e}")
        sys.exit(1)


# ------------------------------------------------------------
# Session 3.5: System boundaries and input validation
# Why: the LLM should not decide what the program already knows.
# ------------------------------------------------------------
def validate_user_input(user_input: str) -> Tuple[bool, Optional[str]]:
    """
    Validate user input BEFORE sending anything to the model.
    Returning (ok, error_message).
    """
    if not user_input.strip():
        return False, "Please enter a message."
    if len(user_input) > MAX_USER_CHARS:
        return False, f"Message too long (max {MAX_USER_CHARS} characters)."
    return True, None


def is_command(user_input: str) -> bool:
    return user_input.lower() in {"quit", "exit", "reset"}


def handle_command(user_input: str, state: ConversationState) -> Optional[str]:
    """
    Handle commands in code (not in the LLM).
    Why: control flow belongs to the application.
    Returns a string response to print, or None if command means 'exit'.
    """
    cmd = user_input.lower()
    if cmd in {"quit", "exit"}:
        return None
    if cmd == "reset":
        state.clear()
        return "🔄 Conversation history cleared."
    return "Unknown command."


# ------------------------------------------------------------
# Session 3.5: Prompt construction functions (prompts are built, not written ad hoc)
# Why: reduces drift, increases inspectability, and makes behaviour testable.
# ------------------------------------------------------------
def build_initial_messages(pdf_text: str, user_question: str) -> List[ChatMessage]:
    """
    First turn: we provide the document as context and ask the user’s question.
    This is the “prompt interface” for initial grounding.
    """
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=(
                f"Here is the document:\n\n{pdf_text}\n\n---\n\n"
                f"Now, my question: {user_question}"
            ),
        ),
    ]


def build_followup_messages(state: ConversationState, user_question: str) -> List[ChatMessage]:
    """
    Follow-up turns: we rely on chat history.
    Notice we keep the system instruction stable and simple.
    """
    # We keep a stable system message in history to anchor behaviour.
    return state.history + [ChatMessage(role="user", content=user_question)]


# ------------------------------------------------------------
# Session 3.5: Failure isolation (guarded model call)
# Why: failures should be contained; users should see safe behaviour.
# ------------------------------------------------------------
def safe_chat(llm, messages: List[ChatMessage]) -> str:
    """
    Wrap LLM call so exceptions don’t take down the app.
    In production you would log error details and maybe retry.
    """
    try:
        response = llm.chat(messages)
        if not response or not str(response).strip():
            raise ValueError("Empty model response")
        return response.strip()
    except Exception as e:
        log.error(f"LLM call failed: {e}", exc_info=True)
        return "Sorry — I had trouble generating a response just now."


# ------------------------------------------------------------
# Session 3.5: State update policy
# Why: keep history clean and predictable; avoid storing huge doc repeatedly.
# ------------------------------------------------------------
def update_state_after_first_turn(
    state: ConversationState,
    user_question: str,
    assistant_response: str
) -> None:
    """
    After the first turn, we DO NOT keep the full document in history.
    Why: it bloats context and makes later turns unreliable.
    Instead we store a stable system message + the user question + the answer.
    """
    state.clear()
    state.add(ChatMessage(role="system", content=FOLLOWUP_SYSTEM_PROMPT))
    state.add(ChatMessage(role="user", content=f"[Document context provided]\n\n{user_question}"))
    state.add(ChatMessage(role="assistant", content=assistant_response))


def update_state_after_followup(
    state: ConversationState,
    user_question: str,
    assistant_response: str
) -> None:
    """
    For follow-ups, store only the conversational exchange.
    """
    state.add(ChatMessage(role="user", content=user_question))
    state.add(ChatMessage(role="assistant", content=assistant_response))


# ------------------------------------------------------------
# Main application loop (kept familiar; internals now structured and teachable)
# ------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Chat with a PDF using your local LLM")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=DEFAULT_MAX_DOC_CHARS,
        help="Maximum characters extracted from the PDF (kept small for local model context)",
    )
    args = parser.parse_args()

    # Load PDF (unchanged conceptually from Session 3)
    print(f"\n📄 Loading PDF: {args.pdf_path}")
    pdf_text = extract_pdf_text(args.pdf_path, max_chars=args.max_doc_chars)
    print(f"✓ Loaded {len(pdf_text)} characters\n")

    # Initialise LLM
    # Why load settings at all? It shows there is an app config surface,
    # even if this script doesn’t use every setting yet.
    _settings = load_settings()
    llm = get_llm()

    # Session 3.5: explicit state object replaces raw list
    state = ConversationState()

    print("=" * 70)
    print("💬 PDF CHATBOT (Session 3.5 Refactor)")
    print("=" * 70)
    print(f"\nPrompt: {PROMPT_NAME} {PROMPT_VERSION}")
    print("\nThe PDF content is loaded. Ask me anything about it!")
    print("Commands: 'quit'/'exit' to end, 'reset' to clear history\n")

    while True:
        try:
            user_input = input("You: ").strip()

            # System boundary: handle commands in code, not in the model
            if is_command(user_input):
                cmd_result = handle_command(user_input, state)
                if cmd_result is None:
                    print("\n👋 Goodbye!\n")
                    break
                print(f"\n{cmd_result}\n")
                continue

            ok, error_msg = validate_user_input(user_input)
            if not ok:
                print(f"\n⚠️  {error_msg}\n")
                continue

            # Build messages differently depending on whether this is the first question
            if not state.has_history():
                messages = build_initial_messages(pdf_text=pdf_text, user_question=user_input)
            else:
                messages = build_followup_messages(state=state, user_question=user_input)

            print("\n🤖 Assistant: ", end="", flush=True)
            response = safe_chat(llm, messages)
            print(response + "\n")

            # Update state (policy differs for first turn vs follow-ups)
            if not state.has_history():
                update_state_after_first_turn(state, user_input, response)
            else:
                update_state_after_followup(state, user_input, response)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            log.error(f"Chat error: {e}", exc_info=True)


if __name__ == "__main__":
    main()