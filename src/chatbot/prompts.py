"""
Session 4: Prompt Management
Session 3: Demonstrates prompt versioning and structure
"""

from __future__ import annotations

from pathlib import Path
from utils.prompt_loader import load_prompt

# Session 3: Prompt versioning
# Prompts are loaded from files for easy iteration and version control

def get_chatbot_v1_prompt() -> str:
    """Load v1.0 system prompt (simple version)"""
    return load_prompt("chatbot_v1_system.txt")


def get_chatbot_v2_prompt() -> str:
    """Load v2.0 system prompt (structured version)"""
    return load_prompt("chatbot_v2_system.txt")


# Default prompt for the service
def get_default_prompt() -> str:
    """Get default chatbot system prompt"""
    return get_chatbot_v2_prompt()
