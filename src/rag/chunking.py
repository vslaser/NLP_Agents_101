from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    meta: dict


def size_based_chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150,
    base_meta: dict | None = None,
) -> list[Chunk]:
    """Default chunking: fixed-size with character overlap.

    - Simple, predictable, and usually good enough to start.
    - Overlap reduces boundary-loss for answers spanning chunk edges.
    """
    text = text.replace("\r\n", "\n")
    meta = base_meta or {}
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_txt = text[start:end].strip()
        if chunk_txt:
            chunks.append(Chunk(text=chunk_txt, meta={**meta, "chunk_index": idx}))
            idx += 1
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def semantic_chunk_text(
    text: str,
    max_chars: int = 900,
    overlap_paragraphs: int = 1,
    base_meta: dict | None = None,
) -> list[Chunk]:
    """A lightweight semantic-ish chunker.

    Strategy:
    - split by paragraphs (blank lines)
    - pack paragraphs into chunks up to max_chars
    - overlap by repeating the last N paragraphs in the next chunk

    This avoids splitting sentences mid-paragraph and tends to preserve local meaning.
    """
    text = text.replace("\r\n", "\n").strip()
    meta = base_meta or {}
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []

    current: list[str] = []
    current_len = 0
    idx = 0

    def flush():
        nonlocal current, current_len, idx
        if not current:
            return
        joined = "\n\n".join(current).strip()
        if joined:
            chunks.append(Chunk(text=joined, meta={**meta, "chunk_index": idx}))
            idx += 1

    i = 0
    while i < len(paras):
        p = paras[i]
        # if a single paragraph is huge, fall back to slicing it
        if len(p) > max_chars:
            flush()
            # slice big paragraph into max_chars windows
            s = 0
            local = 0
            while s < len(p):
                e = min(len(p), s + max_chars)
                part = p[s:e].strip()
                if part:
                    chunks.append(Chunk(text=part, meta={**meta, "chunk_index": idx, "paragraph_part": local}))
                    idx += 1
                    local += 1
                if e == len(p):
                    break
                s = e
            # start next chunk with overlap from last paras
            current = []
            current_len = 0
            i += 1
            continue

        if current_len + len(p) + (2 if current else 0) <= max_chars:
            current.append(p)
            current_len += len(p) + (2 if current else 0)
            i += 1
        else:
            flush()
            # overlap: keep last N paragraphs
            if overlap_paragraphs > 0:
                current = current[-overlap_paragraphs:]
                current_len = sum(len(x) for x in current) + max(0, len(current) - 1) * 2
            else:
                current = []
                current_len = 0

    flush()
    return chunks


def chunk_text(
    text: str,
    method: str = "size",
    chunk_size: int = 900,
    overlap: int = 150,
    base_meta: dict | None = None,
) -> list[Chunk]:
    """Unified chunker with a default size-based strategy."""
    method = (method or "size").lower().strip()
    if method in {"size", "fixed", "char"}:
        return size_based_chunk_text(text, chunk_size=chunk_size, overlap=overlap, base_meta=base_meta)
    if method in {"semantic", "para", "paragraph"}:
        return semantic_chunk_text(text, max_chars=chunk_size, overlap_paragraphs=max(0, overlap // 150), base_meta=base_meta)
    raise ValueError(f"Unknown chunking method: {method}")
