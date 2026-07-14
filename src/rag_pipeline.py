"""FinAlignRAG — RAG Pipeline (Inference & Prompt Coordination).

Coordinates end-to-end inference across the five ablation systems:

  System name               Retrieval              Model weights
  ──────────────────────────────────────────────────────────────
  base_no_rag               none                   base model
  base_simple_rag           dense only             base model
  base_two_stage_rag        dense + cross-encoder  base model
  sft_two_stage_rag         dense + cross-encoder  base + SFT adapter
  sft_dpo_two_stage_rag     dense + cross-encoder  base + DPO adapter

All five systems use the same ChatML prompt format and output schema as the
SFT/DPO training data, enabling apples-to-apples scoring by eval_harness.py.

HARDWARE TARGET — NVIDIA Titan X (Pascal, sm_61, 12 GB VRAM)
-------------------------------------------------------------
Same constraints as alignment.py: 4-bit NF4, float32 activations, eager attention (no
FlashAttention), CUDA required. RAG paths use GPU FAISS (retrieval_engine.py).
Do NOT call merge_and_unload() on a 4-bit quantized PEFT model — the adapter
layers run alongside the frozen quantized base.

Public API
----------
``RAGPipeline(system_name, config, adapter_path=None)``
    ``setup_retrieval(chunks)``            build GPU FAISS index from chunk list
    ``load_index(path)``                   restore a pre-built index from disk
    ``save_index(path)``                   persist the current index to disk
    ``predict_one(question, ...) -> dict`` full pipeline for one sample
    ``run_predictions(questions_path, output_path) -> list[dict]``

CLI
---
``python -m src.rag_pipeline \\
    --system base_two_stage_rag \\
    --config configs/default.yaml \\
    --chunks data/processed/sft_chunks.jsonl \\
    --questions data/processed/test.jsonl \\
    --output outputs/predictions.jsonl``

Add ``--adapter outputs/sft_adapter`` (or ``outputs/dpo_adapter``) for the
SFT / DPO variants. Use ``--index-dir`` to skip re-embedding on repeat runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any

import faiss
import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from src.retrieval_engine import FinancialRetrievalEngine

logger = logging.getLogger("finalignrag.rag_pipeline")

# ---------------------------------------------------------------------------
# System registry
# ---------------------------------------------------------------------------
VALID_SYSTEMS: frozenset[str] = frozenset({
    "base_no_rag",
    "base_simple_rag",
    "base_two_stage_rag",
    "sft_two_stage_rag",
    "sft_dpo_two_stage_rag",
})
_RAG_SYSTEMS: frozenset[str] = frozenset({
    "base_simple_rag",
    "base_two_stage_rag",
    "sft_two_stage_rag",
    "sft_dpo_two_stage_rag",
})
_TWO_STAGE_SYSTEMS: frozenset[str] = frozenset({
    "base_two_stage_rag",
    "sft_two_stage_rag",
    "sft_dpo_two_stage_rag",
})

# ---------------------------------------------------------------------------
# Prompt constants (must match alignment.py _SYSTEM_PROMPT exactly)
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a meticulous financial analyst. Answer the question using ONLY the "
    "provided context. Respond with a single valid JSON object with keys: "
    '"answer", "calculation", "evidence", "confidence", "insufficient_context". '
    "If the context does not contain enough information, set "
    '"insufficient_context" to true and do not fabricate numbers.'
)

# Used only for the no-RAG baseline where no context is provided.
_NO_CONTEXT_SYSTEM_PROMPT = (
    "You are a meticulous financial analyst. Answer the question from your "
    "parametric knowledge. Respond with a single valid JSON object with keys: "
    '"answer", "calculation", "evidence", "confidence", "insufficient_context". '
    "Set \"insufficient_context\" to true if you are not confident in the answer."
)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def _load_config(path: str) -> dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Model loading (mirrors alignment.py)
# ---------------------------------------------------------------------------
def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "RAGPipeline requires a CUDA GPU (4-bit QLoRA via bitsandbytes). "
            "torch.cuda.is_available() returned False. "
            "Run on the Titan X Pascal box."
        )


def _bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 with fp32 compute dtype — must match alignment.py to avoid QK^T overflow."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )


def _load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(
    model_name: str,
    adapter_path: str | None,
) -> AutoModelForCausalLM:
    """Load a 4-bit NF4 base model and optionally a PEFT adapter.

    Do NOT call merge_and_unload(): bitsandbytes 4-bit quantized weights cannot
    be merged with LoRA weights in-place. The adapter runs alongside the frozen
    quantized base, which is both correct and memory-efficient.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=_bnb_config(),
        device_map={"": 0},
        attn_implementation="eager",   # no FlashAttention on Pascal sm_61
        torch_dtype=torch.float32,     # fp32 activations — QK^T overflows fp16 on Pascal
        trust_remote_code=True,
    )
    model.config.use_cache = True      # enabled for inference (unlike training)
    if adapter_path:
        logger.info("Loading PEFT adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Prompt helpers (ChatML format — must match alignment.py _build_prompt)
# ---------------------------------------------------------------------------
def _build_prompt(context: str, question: str) -> str:
    """ChatML prompt (system + user) ending at the assistant turn opening.

    When ``context`` is non-empty the RAG system prompt is used; otherwise
    the no-context prompt is used (no-RAG baseline or empty retrieval result).
    """
    if context:
        system = _SYSTEM_PROMPT
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        system = _NO_CONTEXT_SYSTEM_PROMPT
        user_content = f"Question: {question}"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _extract_json_str(raw: str) -> str:
    """Extract the JSON object from raw decoded model output.

    Strips the assistant end token, then finds the outermost ``{...}`` block.
    Returns the substring as-is (even if invalid JSON); eval_harness flags it.
    """
    text = raw.split("<|im_end|>")[0].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end]
    return text


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------
class RAGPipeline:
    """End-to-end inference pipeline for the FinAlignRAG ablation study.

    Parameters
    ----------
    system_name:
        One of the five ablation system names (see ``VALID_SYSTEMS``).
    config:
        Parsed ``configs/default.yaml`` dict returned by :func:`_load_config`.
    adapter_path:
        Path to a PEFT adapter directory (SFT or DPO). Required for the
        ``sft_*`` system names; ignored for base-model variants.
    """

    def __init__(
        self,
        system_name: str,
        config: dict[str, Any],
        adapter_path: str | None = None,
    ) -> None:
        if system_name not in VALID_SYSTEMS:
            raise ValueError(
                f"Unknown system_name {system_name!r}. "
                f"Valid options: {sorted(VALID_SYSTEMS)}"
            )
        _require_cuda()

        self.system_name = system_name
        retrieval_cfg = config.get("retrieval", {})
        self._top_k_dense: int = int(retrieval_cfg.get("top_k_dense", 15))
        self._top_k_rerank: int = int(retrieval_cfg.get("top_k_rerank", 5))
        training_cfg = config.get("training", {}) or {}
        # Use inference_max_seq_length (larger) so RAG prompts with retrieved chunks fit.
        # Falls back to max_seq_length, then 2048.
        self._max_seq_length: int = int(
            training_cfg.get("inference_max_seq_length")
            or training_cfg.get("max_seq_length")
            or 2048
        )

        models_cfg = config.get("models", {})
        model_name: str = models_cfg.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
        emb_name: str = models_cfg.get("embedding_model", "BAAI/bge-large-en-v1.5")
        reranker_name: str = models_cfg.get(
            "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        logger.info("Loading tokenizer: %s", model_name)
        self.tokenizer = _load_tokenizer(model_name)

        logger.info("Loading model [%s] adapter=%s", system_name, adapter_path)
        self.model = _load_model(model_name, adapter_path)

        self.engine: FinancialRetrievalEngine | None = None
        if system_name in _RAG_SYSTEMS:
            logger.info(
                "Initializing retrieval engine (embedder=%s, reranker=%s)",
                emb_name,
                reranker_name,
            )
            self.engine = FinancialRetrievalEngine(
                embedding_model_name=emb_name,
                reranker_model_name=reranker_name,
            )

    # ------------------------------------------------------------------ #
    # Index management
    # ------------------------------------------------------------------ #
    def setup_retrieval(self, chunks: list[dict[str, Any]]) -> None:
        """Encode ``chunks`` and build the GPU FAISS index."""
        self._require_engine("setup_retrieval")
        assert self.engine is not None
        self.engine.index_chunks(chunks)
        logger.info("Retrieval index built: %d chunks.", len(chunks))

    def load_index(self, path: str) -> None:
        """Restore a pre-built FAISS index + metadata from ``path``."""
        self._require_engine("load_index")
        assert self.engine is not None
        self.engine.load_index(path)

    def save_index(self, path: str) -> None:
        """Persist the current FAISS index to ``path``."""
        self._require_engine("save_index")
        assert self.engine is not None
        self.engine.save_index(path)

    def _require_engine(self, method: str) -> None:
        if self.engine is None:
            raise RuntimeError(
                f"{method}() is not available for system '{self.system_name}' "
                "(no retrieval engine). Use a RAG-enabled system name."
            )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    def _retrieve(self, question: str) -> list[dict[str, Any]]:
        """Return retrieved chunks according to the system's retrieval mode."""
        if self.system_name not in _RAG_SYSTEMS or self.engine is None:
            return []
        if self.system_name in _TWO_STAGE_SYSTEMS:
            return self.engine.query(
                question,
                top_k_dense=self._top_k_dense,
                top_k_rerank=self._top_k_rerank,
            )
        return self._dense_only_retrieve(question)

    def _dense_only_retrieve(self, question: str) -> list[dict[str, Any]]:
        """Dense-only retrieval: FAISS search without the cross-encoder.

        Used by the ``base_simple_rag`` baseline to isolate the reranker's
        contribution. Queries the same GPU FAISS index as two-stage RAG but
        returns results ordered by cosine similarity (``dense_score``).
        """
        engine = self.engine
        assert engine is not None and engine.index is not None, (
            "_dense_only_retrieve() called before the index was built/loaded."
        )
        query_vec = engine.embedder.encode(
            [question],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        query_vec = np.ascontiguousarray(query_vec, dtype="float32")
        faiss.normalize_L2(query_vec)

        k = min(self._top_k_rerank, len(engine._chunks))
        scores, indices = engine.index.search(query_vec, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = dict(engine._chunks[idx])
            chunk["dense_score"] = float(score)
            results.append(chunk)
        return results

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #
    def _build_context(self, chunks: list[dict[str, Any]], question: str, max_new_tokens: int = 768) -> str:
        """Join retrieved chunk texts, trimming chunks that would overflow the token budget.

        Trims from the END of the chunk list (lowest-ranked chunks first) so that
        the question and assistant marker are always preserved in the final prompt.
        The prompt overhead (system + user wrapper + question) is ~200 tokens;
        budget = max_seq_length - max_new_tokens - 200 tokens for context.
        """
        budget_tokens = self._max_seq_length - max_new_tokens - 200
        sep = "\n\n---\n\n"
        kept: list[str] = []
        used = 0
        for chunk in chunks:
            # Rough token estimate: 1 token ≈ 4 chars for English financial text
            chunk_tokens = len(chunk["text"]) // 4
            if used + chunk_tokens > budget_tokens and kept:
                break
            kept.append(chunk["text"])
            used += chunk_tokens
        return sep.join(kept)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _generate(self, prompt: str, max_new_tokens: int = 768) -> str:
        """Tokenize ``prompt``, generate with greedy decoding, decode new tokens."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_seq_length - max_new_tokens,
        ).to("cuda")

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,            # greedy — deterministic for ablation eval
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=False)

    # ------------------------------------------------------------------ #
    # Public prediction API
    # ------------------------------------------------------------------ #
    def predict_one(
        self,
        question: str,
        sample_id: str = "",
        ground_truth_answer: str = "",
        should_refuse: bool = False,
    ) -> dict[str, Any]:
        """Run the full pipeline for one question; return a prediction record.

        The returned dict matches the Prediction JSONL schema consumed by
        ``eval_harness.evaluate_prediction_file``.
        """
        t0 = time.perf_counter()

        chunks = self._retrieve(question)
        context = self._build_context(chunks, question) if chunks else ""
        prompt = _build_prompt(context, question)
        raw_output = self._generate(prompt)
        predicted_json = _extract_json_str(raw_output)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "id": sample_id,
            "system_name": self.system_name,
            "question": question,
            "ground_truth_answer": ground_truth_answer,
            "predicted_json": predicted_json,
            "retrieved_chunks": chunks,
            "should_refuse": should_refuse,
            "latency_ms": round(latency_ms, 1),
        }

    def run_predictions(
        self,
        questions_path: str,
        output_path: str,
    ) -> list[dict[str, Any]]:
        """Predict every record in ``questions_path`` and write to ``output_path``.

        Input JSONL per-line schema::

            {
              "id": "sample_001",           // optional; defaults to line index
              "question": "...",
              "ground_truth_answer": "...", // optional
              "should_refuse": false        // optional; default false
            }

        Returns the list of prediction dicts.
        """
        records = _load_jsonl(questions_path)
        predictions: list[dict[str, Any]] = []

        for i, rec in enumerate(records):
            sample_id = str(rec.get("id") or i)
            question = rec.get("question", "")
            if not question:
                logger.warning("Record %s missing 'question' — skipping.", sample_id)
                continue

            pred = self.predict_one(
                question=question,
                sample_id=sample_id,
                ground_truth_answer=str(rec.get("ground_truth_answer", "")),
                should_refuse=bool(rec.get("should_refuse", False)),
            )
            relevant_chunk_ids = (
                rec.get("relevant_chunk_ids") or rec.get("gold_chunk_ids")
            )
            if relevant_chunk_ids is not None:
                pred["relevant_chunk_ids"] = relevant_chunk_ids
            predictions.append(pred)

            if (i + 1) % 10 == 0:
                logger.info("Processed %d / %d", i + 1, len(records))

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for pred in predictions:
                fh.write(json.dumps(pred, ensure_ascii=False) + "\n")

        logger.info("Wrote %d predictions to %s", len(predictions), output_path)
        return predictions


# ---------------------------------------------------------------------------
# Shared JSONL loader (used by CLI and RAGPipeline)
# ---------------------------------------------------------------------------
def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON on line %d of %s", lineno, path)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.rag_pipeline",
        description=(
            "FinAlignRAG inference: retrieve → prompt → generate → "
            "write prediction JSONL for eval_harness.py."
        ),
    )
    parser.add_argument(
        "--system",
        required=True,
        choices=sorted(VALID_SYSTEMS),
        help="Ablation system to run.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/default.yaml.",
    )
    parser.add_argument(
        "--questions",
        required=True,
        help=(
            "Input JSONL with question records "
            "(fields: id, question, ground_truth_answer, should_refuse)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for prediction JSONL.",
    )
    parser.add_argument(
        "--chunks",
        default=None,
        help=(
            "Chunks JSONL (e.g. data/processed/sft_chunks.jsonl) to build the "
            "retrieval index from. Required for RAG systems unless --index-dir is set."
        ),
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Load a pre-built FAISS index from this directory (skips re-embedding).",
    )
    parser.add_argument(
        "--save-index",
        default=None,
        help="After building from --chunks, save the index here for future reuse.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to a PEFT adapter directory (SFT or DPO).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    set_seed(args.seed)

    config = _load_config(args.config)
    pipeline = RAGPipeline(
        system_name=args.system,
        config=config,
        adapter_path=args.adapter,
    )

    if args.system in _RAG_SYSTEMS:
        if args.index_dir:
            pipeline.load_index(args.index_dir)
        elif args.chunks:
            chunks = _load_jsonl(args.chunks)
            pipeline.setup_retrieval(chunks)
            if args.save_index:
                pipeline.save_index(args.save_index)
        else:
            parser.error(
                f"--system {args.system!r} requires --chunks or --index-dir."
            )

    pipeline.run_predictions(args.questions, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
