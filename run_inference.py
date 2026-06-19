"""Run inference for all five ablation systems and write per-system prediction files.

Systems (in order):
  1. base_no_rag               — base model, no retrieval
  2. base_simple_rag           — base model, dense-only retrieval
  3. base_two_stage_rag        — base model, dense + cross-encoder
  4. sft_two_stage_rag         — SFT adapter, dense + cross-encoder
  5. sft_dpo_two_stage_rag     — DPO adapter, dense + cross-encoder

The FAISS index is built once from data/processed/chunks.jsonl and saved to
outputs/faiss_index/ so subsequent RAG systems load it instantly.

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_inference.py
  CUDA_VISIBLE_DEVICES=0 python run_inference.py --systems sft_two_stage_rag sft_dpo_two_stage_rag
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time

import torch

from src.rag_pipeline import (
    VALID_SYSTEMS,
    _load_config,
    _load_jsonl,
    RAGPipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_inference")

CONFIG       = "configs/default.yaml"
CHUNKS       = "data/processed/chunks.jsonl"
QUESTIONS    = "data/processed/questions.jsonl"
INDEX_DIR    = "outputs/faiss_index"
PREDICTIONS  = "outputs/predictions"
SFT_ADAPTER  = "outputs/sft_adapter"
DPO_ADAPTER  = "outputs/dpo_adapter"

SYSTEM_ORDER = [
    "base_no_rag",
    "base_simple_rag",
    "base_two_stage_rag",
    "sft_two_stage_rag",
    "sft_dpo_two_stage_rag",
]

ADAPTER_MAP: dict[str, str | None] = {
    "base_no_rag":           None,
    "base_simple_rag":       None,
    "base_two_stage_rag":    None,
    "sft_two_stage_rag":     SFT_ADAPTER,
    "sft_dpo_two_stage_rag": DPO_ADAPTER,
}

RAG_SYSTEMS = {"base_simple_rag", "base_two_stage_rag", "sft_two_stage_rag", "sft_dpo_two_stage_rag"}


def run_system(system: str, config: dict, index_built: bool) -> bool:
    """Run one ablation system. Returns True if index was built during this call."""
    out_path = os.path.join(PREDICTIONS, f"{system}.jsonl")
    if os.path.exists(out_path):
        logger.info("[%s] prediction file already exists — skipping.", system)
        return index_built

    logger.info("=" * 60)
    logger.info("Running system: %s", system)
    logger.info("=" * 60)

    adapter = ADAPTER_MAP[system]
    pipeline = RAGPipeline(system_name=system, config=config, adapter_path=adapter)

    if system in RAG_SYSTEMS:
        if index_built and os.path.exists(INDEX_DIR):
            logger.info("Loading pre-built FAISS index from %s", INDEX_DIR)
            pipeline.load_index(INDEX_DIR)
        else:
            logger.info("Building FAISS index from %s …", CHUNKS)
            chunks = _load_jsonl(CHUNKS)
            pipeline.setup_retrieval(chunks)
            os.makedirs(INDEX_DIR, exist_ok=True)
            pipeline.save_index(INDEX_DIR)
            logger.info("Index saved to %s", INDEX_DIR)
            index_built = True
    else:
        logger.info("No retrieval for %s", system)

    os.makedirs(PREDICTIONS, exist_ok=True)
    t0 = time.perf_counter()
    pipeline.run_predictions(QUESTIONS, out_path)
    elapsed = time.perf_counter() - t0
    logger.info("[%s] Done — %.0f s total, saved to %s", system, elapsed, out_path)

    # Free GPU memory before loading the next model
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return index_built


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=sorted(VALID_SYSTEMS),
        default=SYSTEM_ORDER,
        help="Subset of systems to run (default: all five in order).",
    )
    args = parser.parse_args()

    config = _load_config(CONFIG)
    index_built = os.path.exists(INDEX_DIR)

    for system in args.systems:
        index_built = run_system(system, config, index_built)

    logger.info("All systems complete. Predictions in %s/", PREDICTIONS)


if __name__ == "__main__":
    main()
