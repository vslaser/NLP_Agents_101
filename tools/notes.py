from __future__ import annotations
from langchain_core.tools import tool
from utils.db import get_conn

@tool
def create_note(title: str, body: str) -> str:
    """Create a note in the notes database."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO notes(title, body) VALUES(?, ?)", (title, body))
    conn.commit()
    note_id = cur.lastrowid
    return f"Saved note #{note_id}: {title}"

@tool
def list_notes(limit: int = 10) -> str:
    """List most recent notes."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, title, body, created_at FROM notes ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    if not rows:
        return "No notes yet."
    lines = []
    for rid, title, body, created_at in rows:
        lines.append(f"- #{rid} [{created_at}] {title}: {body}")
    return "\n".join(lines)
