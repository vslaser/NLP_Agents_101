from __future__ import annotations
from utils.messages import ChatMessage

def short_term(messages: list[ChatMessage], keep_last_n: int = 6) -> list[ChatMessage]:
    system = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    return system + rest[-keep_last_n:]
