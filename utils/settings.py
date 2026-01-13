from __future__ import annotations
from pydantic import BaseModel
from dotenv import load_dotenv
import os

class Settings(BaseModel):
    llm_mode: str = "local"

    local_model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    local_max_new_tokens: int = 180
    local_temperature: float = 0.3

    openai_compat_base_url: str = ""
    openai_compat_api_key: str = ""
    openai_compat_model: str = ""

    notes_db_path: str = ".data/notes.sqlite3"

def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        llm_mode=os.getenv("LLM_MODE", "local"),

        local_model_id=os.getenv("LOCAL_MODEL_ID", Settings().local_model_id),
        local_max_new_tokens=int(os.getenv("LOCAL_MAX_NEW_TOKENS", str(Settings().local_max_new_tokens))),
        local_temperature=float(os.getenv("LOCAL_TEMPERATURE", str(Settings().local_temperature))),

        openai_compat_base_url=os.getenv("OPENAI_COMPAT_BASE_URL", ""),
        openai_compat_api_key=os.getenv("OPENAI_COMPAT_API_KEY", ""),
        openai_compat_model=os.getenv("OPENAI_COMPAT_MODEL", ""),

        notes_db_path=os.getenv("NOTES_DB_PATH", Settings().notes_db_path),
    )
