from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from src.rag.chunking import chunk_text, Chunk
from src.rag.embeddings import embed_texts
from src.rag.vectorstore import PickleVectorStore


@dataclass(frozen=True)
class IngestionResult:
    store: PickleVectorStore
    store_path: Path
    total_chunks: int


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, list[dict]]:
    """Extract text per page from a PDF.

    Returns:
      - full_text (joined with page separators)
      - page_metas: list of {page_number, char_start, char_end}
    """
    from io import BytesIO
    reader = PdfReader(BytesIO(pdf_bytes))
    parts: list[str] = []
    metas: list[dict] = []
    cursor = 0
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        page_text = page_text.strip()
        start = cursor
        if page_text:
            parts.append(f"\n\n[PAGE {i+1}]\n\n{page_text}")
            cursor += len(parts[-1])
        end = cursor
        metas.append({"page_number": i + 1, "char_start": start, "char_end": end})
    return "".join(parts).strip(), metas


def ingest_text_to_pickle(
    *,
    raw_text: str,
    store_path: str | Path,
    source_name: str,
    chunk_method: str = "size",
    chunk_size: int = 900,
    overlap: int = 150,
) -> IngestionResult:
    base_meta = {"source": source_name}
    chunks: list[Chunk] = chunk_text(
        raw_text,
        method=chunk_method,
        chunk_size=chunk_size,
        overlap=overlap,
        base_meta=base_meta,
    )
    vectors = embed_texts([c.text for c in chunks])
    store = PickleVectorStore.from_chunks(chunks, vectors)
    saved = store.save(store_path)
    print(f"Ingested {len(chunks)} chunks to {saved}")
    return IngestionResult(store=store, store_path=saved, total_chunks=len(chunks))


def ingest_pdf_to_pickle(
    *,
    pdf_bytes: bytes,
    store_path: str | Path,
    source_name: str,
    chunk_method: str = "size",
    chunk_size: int = 900,
    overlap: int = 150,
) -> IngestionResult:
    text, _page_metas = extract_text_from_pdf(pdf_bytes)
    # page metas are coarse; we also store page markers in text itself
    return ingest_text_to_pickle(
        raw_text=text,
        store_path=store_path,
        source_name=source_name,
        chunk_method=chunk_method,
        chunk_size=chunk_size,
        overlap=overlap,
    )


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Ingest PDF to vector store")
    parser.add_argument("--input", required=True, help="Path to input PDF file")
    parser.add_argument("--store", required=True, help="Path to output vector store")
    parser.add_argument("--chunking", default="size", help="Chunking method (default: size)")
    parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size (default: 900)")
    parser.add_argument("--overlap", type=int, default=150, help="Chunk overlap (default: 150)")
    
    args = parser.parse_args()
    
    # Read the PDF file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    with open(input_path, "rb") as f:
        pdf_bytes = f.read()
    
    # Ingest the PDF
    result = ingest_pdf_to_pickle(
        pdf_bytes=pdf_bytes,
        store_path=args.store,
        source_name=input_path.name,
        chunk_method=args.chunking,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    
    print(f"Successfully ingested {result.total_chunks} chunks")
    print(f"Vector store saved to: {result.store_path}")
