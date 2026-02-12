"""
Session 4: Production Chatbot Web UI
Demonstrates:
- Session 1: Clean module structure and imports
- Session 3: Prompt versioning and impact on behavior
- Session 3.5: State management in web applications
- Session 4: User-facing production interface
"""

from __future__ import annotations

import streamlit as st
from src.chatbot.service import ChatbotService
from src.chatbot.prompts import get_chatbot_v1_prompt, get_chatbot_v2_prompt
from utils.llm import RuntimeLLMConfig
from utils.messages import ChatMessage

st.set_page_config(page_title="LLM Chatbot", page_icon="💬", layout="centered")

# ============================================================================
# Session 4: UI Configuration
# ============================================================================

st.title("💬 LLM-Based Chatbot")
st.caption("Session 4: Production Chatbot Interface")

# Sidebar: Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Session 3: Prompt selection demonstrates impact of prompt engineering
    st.subheader("System Prompt")
    prompt_version = st.selectbox(
        "Version",
        options=["v1.0 (Simple)", "v2.0 (Structured)"],
        help="Session 3: Different prompts produce different behaviors. Try both!"
    )
    
    # Load selected prompt
    system_prompt = get_chatbot_v1_prompt() if "v1.0" in prompt_version else get_chatbot_v2_prompt()
    
    with st.expander("📝 View Current Prompt"):
        st.text(system_prompt)
    
    st.divider()
    
    # Session 4: Runtime LLM parameters
    st.subheader("LLM Parameters")
    temperature = st.slider(
        "Temperature", 
        0.0, 1.0, 0.7, 0.1,
        help="Higher = more creative, Lower = more focused"
    )
    max_tokens = st.slider(
        "Max tokens", 
        64, 512, 256, 32,
        help="Maximum length of response"
    )
    
    st.divider()
    
    # Session 3.5: Memory management
    st.subheader("Memory Settings")
    max_history = st.slider(
        "History (turns)", 
        5, 30, 10, 1,
        help="Number of conversation turns to remember"
    )
    max_input_length = st.slider(
        "Max input length", 
        500, 3000, 2000, 100,
        help="Maximum characters per message"
    )
    
    st.divider()
    
    # Clear chat button
    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    # Session stats
    st.divider()
    st.caption("**Session Statistics**")
    history_count = len(st.session_state.get("chatbot_history", [])) - 1  # Exclude system
    turn_count = max(0, (history_count) // 2)
    st.caption(f"Turns: {turn_count}")
    st.caption(f"Messages: {history_count}")


# ============================================================================
# Session 3.5: Initialize State
# ============================================================================

# Initialize chatbot service with current configuration
# Note: Service is recreated on config change to apply new settings
if "chatbot_service" not in st.session_state or st.session_state.get("last_config") != (system_prompt, max_history, max_input_length):
    st.session_state.chatbot_service = ChatbotService(
        system_prompt=system_prompt,
        max_history_turns=max_history,
        max_input_length=max_input_length
    )
    st.session_state.last_config = (system_prompt, max_history, max_input_length)

# Initialize conversation history
if "chatbot_history" not in st.session_state:
    st.session_state.chatbot_history = []


# ============================================================================
# Session 4: Chat Interface
# ============================================================================

# Display chat history
history = st.session_state.chatbot_history
for msg in history:
    if msg.role == "system":
        continue  # Don't show system prompt to user
    with st.chat_message(msg.role):
        st.markdown(msg.content)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Session 4: Runtime configuration for LLM
                runtime = RuntimeLLMConfig(
                    temperature=temperature,
                    max_new_tokens=max_tokens
                )
                
                # Call service layer (Session 3.5: separation of concerns)
                response, updated_history = st.session_state.chatbot_service.chat(
                    prompt,
                    history,
                    runtime
                )
                
                # Display response
                st.markdown(response)
                
                # Update state
                st.session_state.chatbot_history = updated_history
                
            except ValueError as e:
                # Session 3.5: Handle validation errors
                st.error(f"⚠️ Input validation error: {e}")
                
            except Exception as e:
                # Session 3.5: Failure isolation
                st.error(f"❌ Error: {e}")
                st.caption("The chatbot encountered an error. Please try again or clear the chat.")


# ============================================================================
# Session 4: Help Section
# ============================================================================

with st.expander("ℹ️ About This Chatbot"):
    st.markdown("""
    ### Session 4: Production Chatbot
    
    This chatbot demonstrates key concepts from the course:
    
    **Session 1: Project Structure**
    - Clean module organization (`src/chatbot/`)
    - Reusable utilities and services
    
    **Session 3: Prompt Engineering**
    - Try different prompt versions (v1.0 vs v2.0)
    - See how prompt structure affects behavior
    - Versioned prompts stored as .txt files
    
    **Session 3.5: Application Design**
    - Explicit state management (conversation history)
    - System boundaries (validation before LLM call)
    - Failure isolation (errors don't crash the app)
    - Configuration as data (runtime parameters)
    
    **Session 4: Production Features**
    - Web-based interface with Streamlit
    - Runtime configuration controls
    - Memory management
    - User-friendly error handling
    
    ---
    
    **Try These Experiments:**
    1. Switch between v1.0 and v2.0 prompts - notice the difference?
    2. Adjust temperature - see how it affects creativity
    3. Change history turns - test conversation memory
    4. Ask follow-up questions - see context retention
    """)
