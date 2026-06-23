"""Run inference for all five ablation systems and write per-system prediction files.

Systems (in order):
  1. base_no_rag               — base model, no retrieval
  2. base_simple_rag           — base model, dense-only retrieval
  3. base_two_stage_rag        — base model, dense + cross-encoder
  4. sft_two_stage_rag         — SFT adapter, dense + cross-encoder
  5. sft_dpo_two_stage_rag     — DPO adapter, dense + cross-encoder

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_inference.py
  CUDA_VISIBLE_DEVICES=0 python run_inference.py --systems sft_two_stage_rag
  CUDA_VISIBLE_DEVICES=0 python run_inference.py --v2   # v2 corpus + adapter
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
CONFIG_V2    = "configs/v2.yaml"

# v1 paths (original training run)
CHUNKS_V1    = "data/processed/sft_chunks.jsonl"
INDEX_DIR_V1 = "outputs/sft_faiss_index"
SFT_ADAPTER_V1 = "outputs/sft_adapter"
DPO_ADAPTER_V1 = "outputs/dpo_adapter"

# v2 paths (full FinQA + CoT retraining)
CHUNKS_V2    = "data/processed/sft_chunks_v2.jsonl"
INDEX_DIR_V2 = "outputs/sft_faiss_index_v2"
SFT_ADAPTER_V2 = "outputs/sft_adapter_v2"
DPO_ADAPTER_V2 = "outputs/dpo_adapter_v2"

QUESTIONS    = "data/processed/questions.jsonl"
PREDICTIONS  = "outputs/predictions"

SYSTEM_ORDER = [
    "base_no_rag",
    "base_simple_rag",
    "base_two_stage_rag",
    "sft_two_stage_rag",
    "sft_dpo_two_stage_rag",
]

RAG_SYSTEMS = {"base_simple_rag", "base_two_stage_rag", "sft_two_stage_rag", "sft_dpo_two_stage_rag"}


def run_system(
    system: str,
    config: dict,
    index_built: bool,
    chunks_path: str,
    index_dir: str,
    sft_adapter: str,
    dpo_adapter: str,
    predictions_dir: str,
) -> bool:
    """Run one ablation system. Returns True if the FAISS index was built this call."""
    out_path = os.path.join(predictions_dir, f"{system}.jsonl")
    if os.path.exists(out_path):
        logger.info("[%s] prediction file already exists — skipping.", system)
        return index_built

    logger.info("=" * 60)
    logger.info("Running system: %s", system)
    logger.info("=" * 60)

    adapter_map: dict[str, str | None] = {
        "base_no_rag":           None,
        "base_simple_rag":       None,
        "base_two_stage_rag":    None,
        "sft_two_stage_rag":     sft_adapter,
        "sft_dpo_two_stage_rag": dpo_adapter,
    }

    adapter = adapter_map[system]
    pipeline = RAGPipeline(system_name=system, config=config, adapter_path=adapter)

    if system in RAG_SYSTEMS:
        if index_built and os.path.exists(index_dir):
            logger.info("Loading pre-built FAISS index from %s", index_dir)
            pipeline.load_index(index_dir)
        else:
            logger.info("Building FAISS index from %s …", chunks_path)
            chunks = _load_jsonl(chunks_path)
            pipeline.setup_retrieval(chunks)
            os.makedirs(index_dir, exist_ok=True)
            pipeline.save_index(index_dir)
            logger.info("Index saved to %s", index_dir)
            index_built = True
    else:
        logger.info("No retrieval for %s", system)

    os.makedirs(predictions_dir, exist_ok=True)
    t0 = time.perf_counter()
    pipeline.run_predictions(QUESTIONS, out_path)
    elapsed = time.perf_counter() - t0
    logger.info("[%s] Done — %.0f s total, saved to %s", system, elapsed, out_path)

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
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use v2 corpus, index, and adapters (full FinQA + CoT retraining).",
    )
    args = parser.parse_args()

    if args.v2:
        config = _load_config(CONFIG_V2)
        chunks_path = CHUNKS_V2
        index_dir   = INDEX_DIR_V2
        sft_adapter = SFT_ADAPTER_V2
        dpo_adapter = DPO_ADAPTER_V2
        predictions_dir = os.path.join(PREDICTIONS, "v2")
        logger.info("Running in v2 mode (full FinQA + CoT corpus)")
    else:
        config = _load_config(CONFIG)
        chunks_path = CHUNKS_V1
        index_dir   = INDEX_DIR_V1
        sft_adapter = SFT_ADAPTER_V1
        dpo_adapter = DPO_ADAPTER_V1
        predictions_dir = PREDICTIONS

    index_built = os.path.exists(index_dir)

    for system in args.systems:
        index_built = run_system(
            system, config, index_built,
            chunks_path, index_dir, sft_adapter, dpo_adapter, predictions_dir,
        )

    logger.info("All systems complete. Predictions in %s/", predictions_dir)


if __name__ == "__main__":
    main()
