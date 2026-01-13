from __future__ import annotations
from utils.messages import ChatMessage

def trim_history(messages: list[ChatMessage], keep_last_n: int = 10) -> list[ChatMessage]:
    system = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    return system + rest[-keep_last_n:]
