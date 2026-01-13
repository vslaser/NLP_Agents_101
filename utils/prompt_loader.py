from __future__ import annotations
from pathlib import Path
from utils.exceptions import ConfigError

def prompts_dir() -> Path:
    # repo_root/prompts
    return Path(__file__).resolve().parents[1] / "prompts"

def load_prompt(filename: str) -> str:
    path = prompts_dir() / filename
    if not path.exists():
        raise ConfigError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")
