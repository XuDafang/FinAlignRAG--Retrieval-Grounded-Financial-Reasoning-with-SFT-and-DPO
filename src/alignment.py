"""FinAlignRAG — Step 4: Alignment (QLoRA SFT + DPO, single-GPU).

Two separate training stages using QLoRA (4-bit NF4 + LoRA) on a single GPU:
  * ``run_sft``  — supervised fine-tuning for JSON-schema adherence, calculation
                   formatting and graceful refusal, via TRL ``SFTTrainer``.
  * ``run_dpo``  — preference optimization (reduce arithmetic drift / ungrounded
                   claims), via TRL ``DPOTrainer``, initialized from base+SFT adapter.

HARDWARE TARGET — single NVIDIA Titan X (Pascal, sm_61, 12 GB)
--------------------------------------------------------------------------------
QLoRA (4-bit NF4 quantisation + LoRA adapters) on a single GPU.
  * 4-bit base model: 7B × 0.5 bytes ≈ 3.8 GB.  LoRA adapters: ~134 MB.
    Total GPU ≈ 4 GB, fits comfortably in 12 GB.
  * bitsandbytes NF4 with double-quant; compute dtype fp16 (scalar fp16 on Pascal).
  * gradient checkpointing enabled; batch=1, grad_accum=16.
  * No DeepSpeed / distributed -- plain ``python -m src.alignment``.

Launch (single-GPU):
  CUDA_VISIBLE_DEVICES=0 python -m src.alignment --mode sft \\
      --config configs/default.yaml --data data/sft/train.jsonl
  CUDA_VISIBLE_DEVICES=0 python -m src.alignment --mode dpo \\
      --config configs/default.yaml --data data/dpo/train.jsonl \\
      --sft_adapter outputs/sft_adapter/
Add ``--debug`` to cap training at ``max_steps=5`` for a fast smoke run.

TRL API note: written against TRL ~0.11–0.12 (``SFTConfig``/``DPOConfig`` with
``dataset_text_field``/``max_seq_length``/``beta``). Pin the version
(see requirements.txt) if a newer release renames these.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

logger = logging.getLogger("finalignrag.alignment")

# Default LoRA target modules (Qwen2.5 attention + MLP projections).
_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# System instruction shared by SFT and DPO prompt construction.
_SYSTEM_PROMPT = (
    "You are a meticulous financial analyst. Answer the question using ONLY the "
    "provided context. Respond with a single valid JSON object with keys: "
    '"answer", "calculation", "evidence", "confidence", "insufficient_context". '
    "If the context does not contain enough information, set "
    '"insufficient_context" to true and do not fabricate numbers.'
)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Centralized training hyperparameters (sourced from configs/default.yaml)."""

    # --- Required fields (per playbook Step 4) ---
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "outputs"
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    logging_steps: int = 10
    max_steps: int = 1000
    fp16: bool = True                  # QLoRA compute dtype fp16 on Pascal
    seed: int = 42

    # --- Implementation extras ---
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = _TARGET_MODULES
    gradient_checkpointing: bool = True
    max_seq_length: int = 1024
    dpo_beta: float = 0.1
    dpo_max_steps: int = 400             # fewer steps than SFT; DPO over-trains quickly
    sft_adapter_dir: str = "outputs/sft_adapter"
    dpo_adapter_dir: str = "outputs/dpo_adapter"
    deepspeed_config: str | None = None  # QLoRA uses single GPU; no DeepSpeed needed

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Build a TrainingConfig from configs/default.yaml."""
        import yaml  # local import; only needed when loading config

        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

        models = cfg.get("models", {}) or {}
        training = cfg.get("training", {}) or {}
        dpo = training.get("dpo", {}) or {}

        kwargs: dict[str, Any] = {}
        if models.get("base_model"):
            kwargs["model_name"] = models["base_model"]
        for key in (
            "output_dir", "lora_rank", "lora_alpha", "learning_rate", "batch_size",
            "gradient_accumulation_steps", "logging_steps", "max_steps", "fp16",
            "seed", "lora_dropout", "gradient_checkpointing", "max_seq_length",
            "sft_adapter_dir", "dpo_adapter_dir", "deepspeed_config",
        ):
            if training.get(key) is not None:
                kwargs[key] = training[key]
        if training.get("target_modules"):
            kwargs["target_modules"] = tuple(training["target_modules"])
        if dpo.get("beta") is not None:
            kwargs["dpo_beta"] = dpo["beta"]
        if dpo.get("max_steps") is not None:
            kwargs["dpo_max_steps"] = dpo["max_steps"]
        if "seed" not in kwargs and cfg.get("project", {}).get("seed") is not None:
            kwargs["seed"] = cfg["project"]["seed"]
        # learning_rate may parse as str from YAML scientific notation in edge cases
        if "learning_rate" in kwargs:
            kwargs["learning_rate"] = float(kwargs["learning_rate"])
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Prompt / dataset formatting (pure functions)
# ---------------------------------------------------------------------------
def _to_json_str(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _build_prompt(context: str, question: str) -> str:
    """ChatML prompt (system + user) ending at the assistant turn."""
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nContext:\n{context}\n\nQuestion: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_sft_samples(examples: dict[str, list]) -> dict[str, list]:
    """Convert SFT examples into prompt-completion training text.

    Input batch keys: ``text`` (RAG context), ``question``, ``target_json``.
    Output: ``{"text": [...]}`` — full ChatML prompt + target JSON completion.
    """
    contexts = examples.get("text") or examples.get("context")
    questions = examples["question"]
    targets = examples["target_json"]
    rendered = [
        _build_prompt(ctx, q) + _to_json_str(tgt) + "<|im_end|>"
        for ctx, q, tgt in zip(contexts, questions, targets)
    ]
    return {"text": rendered}


def format_dpo_pairs(examples: dict[str, list]) -> dict[str, list]:
    """Convert DPO examples into TRL-compatible prompt/chosen/rejected fields.

    Input batch keys: ``text`` (RAG context), ``question``, ``chosen``,
    ``rejected`` (the latter two are answer-JSON strings). Output:
    ``{"prompt": [...], "chosen": [...], "rejected": [...]}``.
    """
    contexts = examples.get("text") or examples.get("context")
    questions = examples["question"]
    chosen = examples["chosen"]
    rejected = examples["rejected"]

    prompts, chos, rej = [], [], []
    for ctx, q, c, r in zip(contexts, questions, chosen, rejected):
        prompts.append(_build_prompt(ctx, q))
        chos.append(_to_json_str(c) + "<|im_end|>")
        rej.append(_to_json_str(r) + "<|im_end|>")
    return {"prompt": prompts, "chosen": chos, "rejected": rej}


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
def _load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base_model(config: TrainingConfig) -> AutoModelForCausalLM:
    """Load the base model in 4-bit NF4 QLoRA mode (bitsandbytes).

    4-bit NF4 + double-quant: 7B × ~0.5 bytes ≈ 3.8 GB, fits on a single 12 GB GPU.
    Compute dtype fp16 is used for the dequant matmul path (scalar fp16 on Pascal).
    """
    from transformers import BitsAndBytesConfig

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,  # compute in fp32 inside each 4-bit layer
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_cfg,
        torch_dtype=torch.float32,     # fp32 activations — QK^T on Pascal overflows fp16
        attn_implementation="eager",   # no FlashAttention on Pascal sm_61
        trust_remote_code=True,
    )
    model.config.use_cache = False     # incompatible with gradient checkpointing
    return model


def _lora_config(config: TrainingConfig) -> LoraConfig:
    return LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------
def run_sft(training_config: TrainingConfig, data_path: str) -> None:
    """Run LoRA SFT with TRL ``SFTTrainer`` and DeepSpeed ZeRO-3."""
    set_seed(training_config.seed)

    tokenizer = _load_tokenizer(training_config.model_name)
    model = _load_base_model(training_config)

    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(
        format_sft_samples, batched=True, remove_columns=dataset.column_names
    )

    sft_config = SFTConfig(
        output_dir=training_config.sft_adapter_dir,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        logging_steps=training_config.logging_steps,
        max_steps=training_config.max_steps,
        fp16=False,                    # fp32 activations (torch_dtype=float32); AMP would reintroduce overflow
        bf16=False,
        gradient_checkpointing=training_config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",      # 8-bit paged Adam for QLoRA memory efficiency
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        save_strategy="no",
        report_to="none",
        seed=training_config.seed,
        dataset_text_field="text",
        max_seq_length=training_config.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=_lora_config(training_config),
        tokenizer=tokenizer,
    )
    trainer.train()

    os.makedirs(training_config.sft_adapter_dir, exist_ok=True)
    trainer.save_model(training_config.sft_adapter_dir)
    tokenizer.save_pretrained(training_config.sft_adapter_dir)
    logger.info("Saved SFT adapter to %s", training_config.sft_adapter_dir)


def run_dpo(training_config: TrainingConfig, data_path: str, sft_adapter_path: str) -> None:
    """Run DPO with TRL ``DPOTrainer`` and DeepSpeed ZeRO-3, initialized from base + SFT adapter."""
    set_seed(training_config.seed)

    tokenizer = _load_tokenizer(training_config.model_name)
    model = _load_base_model(training_config)
    # Policy = base + SFT adapter (trainable). With ref_model=None, TRL forms the
    # reference by disabling the adapter, so no separate ref model is loaded.
    model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)

    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(
        format_dpo_pairs, batched=True, remove_columns=dataset.column_names
    )

    dpo_config = DPOConfig(
        output_dir=training_config.dpo_adapter_dir,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        logging_steps=training_config.logging_steps,
        max_steps=training_config.dpo_max_steps,
        fp16=False,                    # fp32 activations (torch_dtype=float32); AMP would reintroduce overflow
        bf16=False,
        gradient_checkpointing=training_config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",      # 8-bit paged Adam for QLoRA memory efficiency
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        save_strategy="no",
        report_to="none",
        seed=training_config.seed,
        beta=training_config.dpo_beta,
        max_length=512,                # DPO inputs are concat(chosen, rejected); cap to reduce activation size
        max_prompt_length=256,
        precompute_ref_log_probs=True, # cache ref logprobs upfront so training only holds the policy model
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()

    os.makedirs(training_config.dpo_adapter_dir, exist_ok=True)
    trainer.save_model(training_config.dpo_adapter_dir)
    tokenizer.save_pretrained(training_config.dpo_adapter_dir)
    logger.info("Saved DPO adapter to %s", training_config.dpo_adapter_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.alignment",
        description="QLoRA SFT / DPO alignment for FinAlignRAG (GPU-only, fp16/Pascal).",
    )
    parser.add_argument("--mode", required=True, choices=["sft", "dpo"])
    parser.add_argument("--config", required=True, help="Path to configs/default.yaml.")
    parser.add_argument("--data", required=True, help="Path to the training JSONL.")
    parser.add_argument(
        "--sft_adapter",
        default=None,
        help="Path to the trained SFT adapter (required for --mode dpo).",
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode: max_steps=5.")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = TrainingConfig.from_yaml(args.config)
    if args.debug:
        config.max_steps = 5
        config.dpo_max_steps = 5
        logger.info("DEBUG mode enabled: max_steps=5, dpo_max_steps=5")

    if args.mode == "sft":
        run_sft(config, args.data)
    else:  # dpo
        if not args.sft_adapter:
            parser.error("--sft_adapter is required for --mode dpo")
        run_dpo(config, args.data, args.sft_adapter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
