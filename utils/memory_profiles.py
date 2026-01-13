from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage
from utils.messages import ChatMessage


def trim_chat_messages(messages: list[ChatMessage], keep_last_n: int, keep_system: bool = True) -> list[ChatMessage]:
    """Trim a list of ChatMessage for short-term memory."""
    if keep_last_n <= 0:
        return [m for m in messages if m.role == "system"] if keep_system else []
    system = [m for m in messages if m.role == "system"] if keep_system else []
    rest = [m for m in messages if m.role != "system"]
    return system + rest[-keep_last_n:]


def chatbot_short_term(messages: list[ChatMessage], keep_last_n: int = 10) -> list[ChatMessage]:
    return trim_chat_messages(messages, keep_last_n=keep_last_n, keep_system=True)


def rag_short_term(messages: list[ChatMessage], keep_last_n: int = 6) -> list[ChatMessage]:
    # RAG tends to work better with a shorter conversational window + retrieved context
    return trim_chat_messages(messages, keep_last_n=keep_last_n, keep_system=True)


def trim_langchain_messages(messages: list[BaseMessage], keep_last_n: int = 10) -> list[BaseMessage]:
    """Trim LangChain messages, preserving a leading SystemMessage if present."""
    if keep_last_n <= 0:
        return [m for m in messages if isinstance(m, SystemMessage)]
    sys = [m for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    return sys + rest[-keep_last_n:]


def single_agent_short_term(messages: list[BaseMessage], keep_last_n: int = 10) -> list[BaseMessage]:
    return trim_langchain_messages(messages, keep_last_n=keep_last_n)


def multi_agent_short_term(messages: list[BaseMessage], keep_last_n: int = 12) -> list[BaseMessage]:
    # multi-agent needs a little more room for tool observations and routing context
    return trim_langchain_messages(messages, keep_last_n=keep_last_n)
