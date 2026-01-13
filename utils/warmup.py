from utils.settings import load_settings
from utils.logger import get_logger

log = get_logger(__name__)

def warmup_llm():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    s = load_settings()
    log.info(f"Downloading / loading LLM: {s.local_model_id}")

    tokenizer = AutoTokenizer.from_pretrained(s.local_model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        s.local_model_id,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    model.eval()
    log.info("LLM ready.")

def warmup_embeddings():
    from sentence_transformers import SentenceTransformer

    log.info("Downloading / loading embedding model: all-MiniLM-L6-v2")
    SentenceTransformer("all-MiniLM-L6-v2")
    log.info("Embedding model ready.")

if __name__ == "__main__":
    warmup_llm()
    warmup_embeddings()
    log.info("All models downloaded and cached.")
