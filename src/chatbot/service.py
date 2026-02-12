"""
Session 4: Chatbot Service Layer
Demonstrates:
- Session 3: Prompt engineering and structure
- Session 3.5: System boundaries, state management, failure isolation
- Session 4: Production-ready chatbot service
"""

from __future__ import annotations

from typing import Tuple, List
from utils.llm import get_llm, RuntimeLLMConfig
from utils.messages import ChatMessage
from utils.prompt_loader import load_prompt
from utils.memory_profiles import chatbot_short_term
from utils.exceptions import ExternalServiceError
from utils.logger import get_logger

log = get_logger(__name__)

# Legacy function for backward compatibility (used by ui/app.py)
def answer(
    user_text: str,
    history: list[ChatMessage],
    runtime: RuntimeLLMConfig | None = None,
    *,
    keep_last_n: int = 10,
) -> tuple[str, list[ChatMessage]]:
    """
    Legacy chatbot answer function
    Kept for compatibility with existing UI code
    """
    system = load_prompt("chatbot_system.txt")
    if not history:
        history = [ChatMessage(role="system", content=system)]
    history.append(ChatMessage(role="user", content=user_text))

    llm = get_llm(runtime)
    msgs = chatbot_short_term(history, keep_last_n=keep_last_n)
    out = llm.chat(msgs)
    history.append(ChatMessage(role="assistant", content=out))
    return out, history


# Session 4: New service-oriented architecture
class ChatbotService:
    """
    Session 3.5: Proper boundaries and failure isolation
    Session 4: Production-ready chatbot service
    """
    
    def __init__(self, system_prompt: str, max_history_turns: int = 10, max_input_length: int = 2000):
        """
        Initialize chatbot service
        
        Args:
            system_prompt: System prompt defining assistant behavior
            max_history_turns: Maximum conversation turns to keep in memory
            max_input_length: Maximum allowed user input length
        """
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.max_input_length = max_input_length
        self.llm = get_llm()
    
    def validate_input(self, text: str) -> Tuple[bool, str]:
        """
        Session 3.5: Input validation at system boundary
        The application decides what's valid, not the LLM
        
        Returns:
            (is_valid, error_message)
        """
        if not text.strip():
            return False, "Empty input"
        if len(text) > self.max_input_length:
            return False, f"Input too long (max {self.max_input_length} chars)"
        return True, ""
    
    def chat(
        self, 
        user_text: str, 
        history: List[ChatMessage],
        runtime: RuntimeLLMConfig | None = None
    ) -> Tuple[str, List[ChatMessage]]:
        """
        Process a chat message and return response
        
        Session 3.5: Failure isolation and state management
        Session 3: Prompt construction
        
        Args:
            user_text: User's message
            history: Conversation history
            runtime: Optional runtime LLM configuration
            
        Returns:
            (assistant_response, updated_history)
            
        Raises:
            ValueError: If input validation fails
            ExternalServiceError: If LLM call fails
        """
        # Session 3.5: Validate at system boundary
        valid, error = self.validate_input(user_text)
        if not valid:
            raise ValueError(error)
        
        # Session 3: Build messages with proper prompt structure
        if not history:
            history = [ChatMessage(role="system", content=self.system_prompt)]
        
        history.append(ChatMessage(role="user", content=user_text))
        
        # Session 3.5: Apply explicit memory policy
        messages_to_send = self._apply_memory_policy(history)
        
        try:
            # Session 3.5: Failure isolation - wrap LLM call
            llm = get_llm(runtime)
            response = llm.chat(messages_to_send)
            
            if not response or not str(response).strip():
                raise ValueError("Empty model response")
            
            history.append(ChatMessage(role="assistant", content=response))
            return response, history
            
        except Exception as e:
            # Session 3.5: Failure isolation - don't let LLM errors crash the app
            log.error(f"LLM call failed: {e}", exc_info=True)
            raise ExternalServiceError(f"Failed to generate response: {e}")
    
    def _apply_memory_policy(self, history: List[ChatMessage]) -> List[ChatMessage]:
        """
        Session 3.5: Explicit memory management
        Keep system message + last N turns to fit context window
        """
        if len(history) <= self.max_history_turns + 1:
            return history
        
        # Keep system message (first) + last N messages
        return [history[0]] + history[-(self.max_history_turns):]
