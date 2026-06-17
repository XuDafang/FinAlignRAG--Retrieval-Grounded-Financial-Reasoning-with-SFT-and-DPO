# FinAlignRAG: Retrieval-Grounded Financial Reasoning with SFT and DPO

A rigorous ablation study measuring how much each layer of the stack — retrieval quality, supervised fine-tuning, and preference alignment — contributes to accurate numerical reasoning over financial documents.

## Objective

Large language models hallucinate financial numbers. This project quantifies whether the problem is better solved by:

1. **Better retrieval** — surfacing the right document chunks before generation
2. **Supervised fine-tuning (SFT)** — teaching a 7B model to produce structured, auditable JSON answers
3. **Direct preference optimization (DPO)** — penalizing arithmetic drift, fabricated evidence, and overconfident refusals

Five systems are evaluated head-to-head on identical test questions drawn from SEC filings (FinQA / ConvFinQA):

| # | System | Retrieval | Model weights |
|---|--------|-----------|---------------|
| 0 | `base_no_rag` | none | Qwen2.5-7B-Instruct (base) |
| 1 | `base_simple_rag` | dense (FAISS cosine) | base |
| 2 | `base_two_stage_rag` | dense + cross-encoder rerank | base |
| 3 | `sft_two_stage_rag` | dense + cross-encoder rerank | base + SFT adapter |
| 4 | `sft_dpo_two_stage_rag` | dense + cross-encoder rerank | base + SFT→DPO adapter |

> **Note on system 4:** The DPO adapter is not applied to the raw base model. DPO training
> (`alignment.py run_dpo`) initializes from the already-trained SFT adapter and continues
> preference-optimizing those weights. The saved `outputs/dpo_adapter/` therefore encodes
> **both** SFT and DPO — it is the SFT adapter further tuned by DPO. Loading it on top of
> the base model gives a model that has been through both training stages.

The marginal gain from each row isolates one factor, giving a clean ablation table.

---

## Architecture

```
Raw SEC Filings (JSONL)
        │
        ▼
┌───────────────────┐
│  data_pipeline.py │  chunk by 512 tokens, split by ticker (no leakage)
└───────────────────┘
        │
        ▼
 data/processed/
  chunks.jsonl          ◄─── used to build the retrieval index
  train.jsonl           ◄─── used to generate SFT / DPO training pairs
  val.jsonl
  test.jsonl

        │                             ┌──────────────────────┐
        ├─────── index_chunks() ─────►│ retrieval_engine.py  │
        │                             │  FAISS IndexFlatIP   │
        │                             │  + cross-encoder     │
        │                             └──────────────────────┘
        │                                       │ top-5 chunks
        │                                       ▼
        │                             ┌──────────────────────┐
        └─────── run_sft / run_dpo ──►│   alignment.py       │
                                      │  QLoRA SFT + DPO     │
                                      │  (Titan X fp16)      │
                                      └──────────────────────┘
                                                │ adapter weights
                                                ▼
                                      ┌──────────────────────┐
                                      │  rag_pipeline.py     │  ◄─ choose system
                                      │  retrieve → prompt   │
                                      │  → generate → JSON   │
                                      └──────────────────────┘
                                                │ predictions.jsonl
                                                ▼
                                      ┌──────────────────────┐
                                      │  eval_harness.py     │
                                      │  JSON validity       │
                                      │  numerical accuracy  │
                                      │  evidence support    │
                                      │  refusal accuracy    │
                                      │  retrieval recall@5  │
                                      └──────────────────────┘
```

---

## Concrete Example

### Input

**Question** (from FinQA / `data/train.json`):

> What was the percentage change in net cash from operating activities from 2008 to 2009?

**Source document excerpt** (from `JKHY/2009/page_28.pdf`):

```
Net cash from operating activities  2009: $206,588   2008: $181,001
```

**Raw document record** (what `data/raw/documents.jsonl` must contain):

```json
{
  "ticker": "JKHY",
  "source_doc_id": "JKHY_2009_page28",
  "text": "Net income $103,102 $104,222 $104,681 ... Net cash from operating activities $206,588 $181,001 $174,247"
}
```

---

### Output

After running the full pipeline, `rag_pipeline.py` writes one line per question to the prediction JSONL:

```json
{
  "id": "train_000",
  "system_name": "sft_dpo_two_stage_rag",
  "question": "What was the percentage change in net cash from operating activities from 2008 to 2009?",
  "ground_truth_answer": "14.1%",
  "predicted_json": "{"answer": "14.1%", "calculation": "(206588 - 181001) / 181001", "evidence": "Net cash from operating activities was $206,588 in 2009 and $181,001 in 2008", "confidence": 0.95, "insufficient_context": false}",
  "retrieved_chunks": [
    {
      "chunk_id": "JKHY_2009_page28_000",
      "text": "Net cash from operating activities $206,588 $181,001 $174,247",
      "ticker": "JKHY",
      "source_doc_id": "JKHY_2009_page28",
      "chunk_index": 0,
      "dense_score": 0.83,
      "rerank_score": 0.96
    }
  ],
  "should_refuse": false,
  "latency_ms": 840.2
}
```

The `predicted_json` field is then scored by `eval_harness.py`:

- **JSON validity** — all five required keys present? → `true`
- **Numerical accuracy** — `14.1%` within 0.1% of gold `14.1%`? → `true`
- **Evidence support** — do operands `206588` and `181001` appear in the evidence? → `true`
- **Refusal accuracy** — `insufficient_context: false` matches `should_refuse: false`? → `true`
- **Retrieval recall@5** — gold chunk `JKHY_2009_page28_000` in top-5? → `1.0`

---

## Hardware Requirements

- **GPUs**: 4 × NVIDIA Titan X (Pascal, sm_61), 12 GB VRAM each (48 GB total)
- **Precision**: `fp16` — Pascal does **not** support `bfloat16`
- **FAISS**: GPU build required (`faiss-gpu` via conda; `faiss-cpu` fails at runtime)
- **FlashAttention**: NOT used — requires Ampere (sm_80+)

**Training strategy — DeepSpeed ZeRO-3 + fp16 (via `DS_ALLOW_DEPRECATED_FP16=1`):**
All four GPUs are used during SFT and DPO training via DeepSpeed ZeRO stage 3.
Pascal (sm_61) is labelled "deprecated fp16" by DeepSpeed (which gates fp16 on
Volta sm_70+) but the GPU computes fp16 correctly. Setting `DS_ALLOW_DEPRECATED_FP16=1`
unblocks the check. ZeRO-3 shards fp16 weights: 7B × 2 bytes = 14 GB ÷ 4 GPUs = 3.5 GB/GPU.
`HfDeepSpeedConfig` is created before `from_pretrained` so Transformers uses
`GatheredParameters` (not naive `load_state_dict`) for ZeRO-3 compatible loading.
Training launches as:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_ALLOW_DEPRECATED_FP16=1 \
torchrun --nproc_per_node=4 -m src.alignment --mode sft ...
```
4-bit quantization is still used in `rag_pipeline.py` for single-GPU inference.

**Inference strategy — single GPU with 4-bit quantization:**
`rag_pipeline.py` loads the model with 4-bit NF4 quantization (bitsandbytes)
on a single GPU, as inference does not benefit from ZeRO-3 parameter sharding.

---

## Installation

```bash
# 1. GPU FAISS must come from conda (PyPI faiss-gpu is Linux/CUDA only, but the
#    retrieval engine requires faiss.StandardGpuResources which conda provides)
conda install -c pytorch -c nvidia faiss-gpu

# 2. PyTorch with CUDA (match your driver version — check with nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install all other dependencies (includes deepspeed for ZeRO-3 training)
pip install -r requirements.txt
```

---

## Step-by-Step: Running the Project

### Step 1 — Prepare Raw Documents

The FinQA/ConvFinQA data in `data/train.json` must be converted to the
pipeline's JSONL format (`ticker`, `source_doc_id`, `text` per line):

```bash
# Example conversion (adapt to your actual FinQA preprocessing script)
python - <<'EOF'
import json, pathlib

records = json.loads(pathlib.Path("data/train.json").read_text())
out = pathlib.Path("data/raw/documents.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w") as fh:
    for r in records:
        ticker = r["filename"].split("/")[0]          # e.g. "JKHY"
        doc_id = r["filename"].replace("/", "_").replace(".pdf", "")
        text   = " ".join(r.get("pre_text", []) + r.get("post_text", []))
        fh.write(json.dumps({"ticker": ticker, "source_doc_id": doc_id, "text": text}) + "\n")
print("Done:", out)
EOF
```

### Step 2 — Data Pipeline (chunk + split by ticker)

```bash
python -m src.data_pipeline \
    --input  data/raw/documents.jsonl \
    --output-dir data/processed \
    --config configs/default.yaml
```

Outputs to `data/processed/`: `chunks.jsonl`, `train.jsonl`, `val.jsonl`, `test.jsonl`.
Tickers are never shared across splits — leakage is impossible by construction.

### Step 3 — Smoke-test the Retrieval Engine

```bash
python -m src.retrieval_engine --log-level INFO
```

Expects a CUDA GPU + GPU FAISS. Prints `SMOKE TEST PASSED` on success.
Optionally exercise save/load:

```bash
python -m src.retrieval_engine --save-dir /tmp/smoke_index
```

### Step 4 — Prepare SFT Training Data

SFT training expects a JSONL where each line has
`text` (retrieved context), `question`, and `target_json`:

```json
{
  "ticker": "JKHY",
  "source_doc_id": "JKHY_2009_page28",
  "text": "<retrieved context chunks concatenated>",
  "question": "What was the percentage change in net cash from operating activities from 2008 to 2009?",
  "target_json": "{"answer": "14.1%", "calculation": "(206588 - 181001) / 181001", "evidence": "Net cash was $206,588 in 2009 and $181,001 in 2008", "confidence": 0.95, "insufficient_context": false}"
}
```

Place the prepared file at `data/sft/train.jsonl`.

### Step 5 — SFT Training (all 4 GPUs via DeepSpeed ZeRO-3)

Training uses `torchrun` to launch one process per GPU. DeepSpeed ZeRO-3 shards
the 7B fp16 model across all 4 GPUs, eliminating the need for 4-bit quantization
during training and increasing throughput significantly.

```bash
# Two environment variables are required for multi-GPU training on Pascal:
#   DS_ALLOW_DEPRECATED_FP16=1  — Pascal fp16 is "deprecated" in DeepSpeed but works fine
#   PYTORCH_CUDA_ALLOC_CONF=... — reduces fragmentation from ZeRO-3 all-gather buffers

# Quick smoke run (5 steps — verifies the pipeline without waiting)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_ALLOW_DEPRECATED_FP16=1 \
torchrun --nproc_per_node=4 -m src.alignment \
    --mode   sft \
    --config configs/default.yaml \
    --data   data/sft/train.jsonl \
    --debug

# Full training (1000 steps)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_ALLOW_DEPRECATED_FP16=1 \
torchrun --nproc_per_node=4 -m src.alignment \
    --mode   sft \
    --config configs/default.yaml \
    --data   data/sft/train.jsonl
```

Adapter saved to `outputs/sft_adapter/`.

### Step 6 — Prepare DPO Training Data

DPO training expects `text` (context), `question`, `chosen` (correct answer JSON),
and `rejected` (flawed answer JSON — arithmetic error, fabricated evidence, etc.):

```json
{
  "text": "<retrieved context>",
  "question": "What was the percentage change in net cash from operating activities?",
  "chosen":   "{"answer": "14.1%", "calculation": "(206588 - 181001) / 181001", ...}",
  "rejected": "{"answer": "13.9%", "calculation": "(206588 - 181001) / 206588", ...}"
}
```

Place the prepared file at `data/dpo/train.jsonl`.

### Step 7 — DPO Training (all 4 GPUs via DeepSpeed ZeRO-3)

```bash
# Quick smoke run
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_ALLOW_DEPRECATED_FP16=1 \
torchrun --nproc_per_node=4 -m src.alignment \
    --mode        dpo \
    --config      configs/default.yaml \
    --data        data/dpo/train.jsonl \
    --sft_adapter outputs/sft_adapter \
    --debug

# Full training
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DS_ALLOW_DEPRECATED_FP16=1 \
torchrun --nproc_per_node=4 -m src.alignment \
    --mode        dpo \
    --config      configs/default.yaml \
    --data        data/dpo/train.jsonl \
    --sft_adapter outputs/sft_adapter
```

Adapter saved to `outputs/dpo_adapter/`.

### Step 8 — Run Inference (all 5 ablation systems)

Build the retrieval index once from Step 2 chunks, then reuse it across runs:

```bash
# System 0: base model only (no retrieval)
python -m src.rag_pipeline \
    --system    base_no_rag \
    --config    configs/default.yaml \
    --questions data/processed/test.jsonl \
    --output    outputs/preds_0_base_no_rag.jsonl

# System 1: dense-only retrieval (builds index, saves for reuse)
python -m src.rag_pipeline \
    --system     base_simple_rag \
    --config     configs/default.yaml \
    --chunks     data/processed/chunks.jsonl \
    --questions  data/processed/test.jsonl \
    --output     outputs/preds_1_base_simple_rag.jsonl \
    --save-index outputs/faiss_index

# System 2: two-stage RAG (loads saved index)
python -m src.rag_pipeline \
    --system    base_two_stage_rag \
    --config    configs/default.yaml \
    --index-dir outputs/faiss_index \
    --questions data/processed/test.jsonl \
    --output    outputs/preds_2_base_two_stage_rag.jsonl

# System 3: SFT model + two-stage RAG
python -m src.rag_pipeline \
    --system    sft_two_stage_rag \
    --config    configs/default.yaml \
    --index-dir outputs/faiss_index \
    --questions data/processed/test.jsonl \
    --output    outputs/preds_3_sft_two_stage_rag.jsonl \
    --adapter   outputs/sft_adapter

# System 4: SFT + DPO model + two-stage RAG
python -m src.rag_pipeline \
    --system    sft_dpo_two_stage_rag \
    --config    configs/default.yaml \
    --index-dir outputs/faiss_index \
    --questions data/processed/test.jsonl \
    --output    outputs/preds_4_sft_dpo_two_stage_rag.jsonl \
    --adapter   outputs/dpo_adapter
```

### Step 9 — Evaluate All Systems

```bash
for i in 0 1 2 3 4; do
  python -m src.eval_harness \
      --predictions outputs/preds_${i}_*.jsonl \
      --report      reports/metrics_system${i}.json
done
```

Or score one file at a time:

```bash
python -m src.eval_harness \
    --predictions outputs/preds_4_sft_dpo_two_stage_rag.jsonl \
    --report      reports/metrics_sft_dpo.json
```

The report prints a summary table and writes `reports/ablation_results.md`.

### Step 10 — Run Tests

```bash
pytest
```

---

## Expected Ablation Results

*(Fill in after running all five systems.)*

| System | Numerical Accuracy | JSON Validity | Evidence Support | Refusal Accuracy | Retrieval Recall@5 | Latency/query |
|---|---|---|---|---|---|---|
| Base model only | — | — | — | — | N/A | — |
| Base + simple RAG | — | — | — | — | — | — |
| Base + two-stage RAG | — | — | — | — | — | — |
| SFT + two-stage RAG | — | — | — | — | — | — |
| SFT + DPO + two-stage RAG | — | — | — | — | — | — |

---

## Repository Layout

```
configs/
  default.yaml          single source of truth for all hyperparameters
data/
  raw/                  input: one JSONL per corpus (ticker, source_doc_id, text)
  processed/            output of data_pipeline: chunks + train/val/test splits
  sft/                  SFT training pairs (text, question, target_json)
  dpo/                  DPO training pairs (text, question, chosen, rejected)
  train.json            raw FinQA training set (needs conversion → data/raw/)
  dev.json              raw FinQA dev set
outputs/
  sft_adapter/          QLoRA SFT weights (saved by alignment.py)
  dpo_adapter/          QLoRA DPO weights (saved by alignment.py)
reports/                evaluation reports (written by eval_harness.py)
src/
  data_pipeline.py      Step 1 — ingestion, chunking, ticker-split
  retrieval_engine.py   Step 2 — GPU FAISS dense + cross-encoder rerank
  eval_harness.py       Step 3 — deterministic scoring (JSON, math, evidence)
  alignment.py          Step 4 — QLoRA SFT and DPO training
  rag_pipeline.py       Step 5 — end-to-end inference for all 5 ablation systems
```
