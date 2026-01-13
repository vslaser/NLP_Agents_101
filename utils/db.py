from __future__ import annotations
import os
import sqlite3
from utils.settings import load_settings

def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def get_conn() -> sqlite3.Connection:
    s = load_settings()
    _ensure_parent(s.notes_db_path)
    conn = sqlite3.connect(s.notes_db_path, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.commit()
    return conn
