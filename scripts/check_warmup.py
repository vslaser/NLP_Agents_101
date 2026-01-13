"""Small utility to check whether warmup ran and whether models are cached.

Usage:
  python scripts/check_warmup.py --check-cache
  python scripts/check_warmup.py --run-warmup
"""

import argparse
import traceback
from pathlib import Path
import sys
import pathlib as _pl

# Ensure the project root is on sys.path so `from utils import ...` works
# whether the script is run from the repo root or directly from another folder.
sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))

from utils import warmup
from utils.settings import load_settings


def find_in_cache(name: str):
    root = Path.home() / ".cache" / "huggingface"
    if not root.exists():
        return []
    name_l = name.lower()
    matches = [p for p in root.rglob("*") if name_l in p.name.lower()]
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches


def check_model_cache(model_id: str) -> bool:
    parts = [p for p in model_id.replace("\\", "/").split("/") if p]
    search_terms = set(parts + [model_id, parts[-1] if parts else model_id])
    found_any = False
    for term in search_terms:
        matches = find_in_cache(term)
        if matches:
            found_any = True
            print(f"Found {len(matches)} entries for '{term}':")
            for m in matches[:10]:
                print("  ", m)
    if not found_any:
        print(f"No cache entries found for '{model_id}' under ~/.cache/huggingface")
    return found_any


def run_warmups():
    try:
        warmup.warmup_llm()
        print("warmup_llm: SUCCESS")
    except Exception as e:
        print("warmup_llm: ERROR", e)
        traceback.print_exc()

    try:
        warmup.warmup_embeddings()
        print("warmup_embeddings: SUCCESS")
    except Exception as e:
        print("warmup_embeddings: ERROR", e)
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Check warmup and model cache")
    parser.add_argument("--run-warmup", action="store_true", help="Run the warmup functions (may download models)")
    parser.add_argument("--check-cache", action="store_true", help="Check the HF cache for model files")
    parser.add_argument("--model", type=str, default=None, help="Override model id to check")
    parser.add_argument("--embeddings", type=str, default="all-MiniLM-L6-v2", help="Embedding model id to check")

    args = parser.parse_args()

    s = load_settings()
    model_id = args.model or s.local_model_id

    if args.check_cache or not args.run_warmup:
        print(f"Checking cache for LLM: {model_id}")
        check_model_cache(model_id)
        print(f"Checking cache for embeddings: {args.embeddings}")
        check_model_cache(args.embeddings)

    if args.run_warmup:
        print("Running warmup (this may download models now)...")
        run_warmups()


if __name__ == "__main__":
    main()
