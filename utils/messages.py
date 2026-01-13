from __future__ import annotations
from pydantic import BaseModel
from typing import Literal

Role = Literal["system", "user", "assistant", "tool"]

class ChatMessage(BaseModel):
    role: Role
    content: str
    name: str | None = None  # tool name when role='tool'
