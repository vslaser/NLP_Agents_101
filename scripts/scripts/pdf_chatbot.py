"""
Simple PDF Chatbot - Practice prompting with PDF content
Loads a PDF file and provides a chat interface to interact with it using the local LLM
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm import get_llm
from utils.settings import load_settings
from utils.messages import ChatMessage
from utils.logger import get_logger

log = get_logger(__name__)


def extract_pdf_text(pdf_path: str, max_chars: int = 6000) -> str:
    """Extract text from PDF file, truncated to fit model context."""
    try:
        import PyPDF2
        
        text_parts = []
        total_chars = 0
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    page_text = f"--- Page {page_num} ---\n{text}"
                    
                    # Check if adding this page would exceed limit
                    if total_chars + len(page_text) > max_chars:
                        # Add partial page if possible
                        remaining = max_chars - total_chars
                        if remaining > 200:  # Only add if there's meaningful space
                            text_parts.append(page_text[:remaining] + "\n\n[... truncated ...]")
                        break
                    
                    text_parts.append(page_text)
                    total_chars += len(page_text)
            
            result = "\n\n".join(text_parts)
            
            # Print truncation info
            if total_chars >= max_chars or page_num < total_pages:
                print(f"⚠️  Document truncated: Using first ~{len(result)} characters from {page_num}/{total_pages} pages")
                print(f"   (Local model context limit: ~2000 tokens / ~8000 characters)\n")
            
            return result
            
    except ImportError:
        print("ERROR: PyPDF2 not installed. Run: pip install PyPDF2")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not read PDF: {e}")
        sys.exit(1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Chat with a PDF using your local LLM")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    args = parser.parse_args()
    
    # Load PDF
    print(f"\n📄 Loading PDF: {args.pdf_path}")
    pdf_text = extract_pdf_text(args.pdf_path)
    print(f"✓ Loaded {len(pdf_text)} characters\n")
    
    # Initialize LLM
    settings = load_settings()
    llm = get_llm()
    
    # Chat loop
    history = []
    
    print("="*70)
    print("💬 PDF CHATBOT")
    print("="*70)
    print("\nThe PDF content is loaded. Ask me anything about it!")
    print("Commands: 'quit' or 'exit' to end, 'reset' to clear history\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye!\n")
                break
            
            if user_input.lower() == 'reset':
                history = []
                print("\n🔄 Conversation history cleared.\n")
                continue
            
            # First message includes the PDF content as context
            if not history:
                messages = [
                    ChatMessage(
                        role="system",
                        content="You are a helpful assistant. The user has provided a PDF document and will ask questions about it. Answer based on the document content provided."
                    ),
                    ChatMessage(
                        role="user",
                        content=f"Here is the document:\n\n{pdf_text}\n\n---\n\nNow, my question: {user_input}"
                    )
                ]
            else:
                # Subsequent messages use history
                messages = history + [ChatMessage(role="user", content=user_input)]
            
            print("\n🤖 Assistant: ", end="", flush=True)
            response = llm.chat(messages)
            print(response + "\n")
            
            # Update history
            if not history:
                # Store the context-aware first exchange
                history = [
                    ChatMessage(role="system", content="You are a helpful assistant discussing a document with the user."),
                    ChatMessage(role="user", content=f"[Document context provided]\n\n{user_input}"),
                    ChatMessage(role="assistant", content=response)
                ]
            else:
                history.append(ChatMessage(role="user", content=user_input))
                history.append(ChatMessage(role="assistant", content=response))
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            log.error(f"Chat error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
