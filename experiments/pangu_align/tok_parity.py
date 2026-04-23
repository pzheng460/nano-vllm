"""Tokenizer parity check: nano-vllm vs vLLM for PanGu."""
import sys
from pathlib import Path

MODEL = "/mnt/data/weights/openPangu-R-72B-2512" if Path("/mnt/data/weights/openPangu-R-72B-2512").exists() else "/mnt/data/peizhen/openPangu-R-72B-2512-3mtp"
PROMPT = "The capital of France is"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1] if len(sys.argv) > 1 else MODEL, trust_remote_code=True)
ids = tok(PROMPT).input_ids if False else tok(PROMPT, add_special_tokens=True).input_ids
print("model:", MODEL)
print("prompt:", repr(PROMPT))
print("input_ids:", ids)
print("decoded:", repr(tok.decode(ids)))
