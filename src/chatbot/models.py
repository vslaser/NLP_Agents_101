"""
Session 4: Data Models for Chatbot
Demonstrates Session 3.5 principle: Explicit configuration and state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from utils.messages import ChatMessage


@dataclass
class ChatbotConfig:
    """
    Session 3.5: Explicit configuration
    Makes chatbot behavior configurable and testable
    """
    system_prompt: str
    max_history_turns: int = 10
    temperature: float = 0.7
    max_tokens: int = 256
    max_input_length: int = 2000


@dataclass
class ConversationContext:
    """
    Session 3.5: Explicit state management
    Replaces implicit state (raw lists) with structured data
    """
    messages: List[ChatMessage] = field(default_factory=list)
    user_id: str = "default_user"
    session_id: str = "default_session"
    turn_count: int = 0
    
    def clear(self) -> None:
        """Reset conversation state"""
        self.messages = []
        self.turn_count = 0
    
    def has_history(self) -> bool:
        """Check if conversation has started"""
        return len(self.messages) > 0
