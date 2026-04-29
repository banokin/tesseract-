from __future__ import annotations

QWEN_30_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct:novita"
LLAMA_4_SCOUT_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct:novita"

TWO_MODELS_MAP: dict[str, str] = {
    "qwen30": QWEN_30_MODEL,
    "llama4scout": LLAMA_4_SCOUT_MODEL,
}
