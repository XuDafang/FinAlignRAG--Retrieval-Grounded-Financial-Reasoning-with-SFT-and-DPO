# FinAlignRAG: Retrieval-Grounded Financial Reasoning with SFT and DPO

An ablation study measuring how much each layer of the stack — retrieval quality, supervised fine-tuning (SFT), and preference alignment (DPO) — contributes to accurate numerical reasoning over financial documents (FinQA / SEC filings).

---

## Skills Demonstrated

**Model Fine-Tuning & Alignment**
- Fine-tuned Qwen2.5-7B on financial QA via QLoRA (4-bit NF4, LoRA rank-16) on a single 12 GB GPU across two training runs (2,109 and 3,439 examples), improving numerical accuracy 0% → 15.3% and refusal accuracy 14.7% → 100% on a 300-question held-out eval set
- Implemented SFT and DPO training pipelines using HuggingFace TRL, PEFT, and bitsandbytes; diagnosed and documented DPO reward collapse caused by trivially-separable synthetic rejections saturating the objective at step 1 with loss ≈ 0
- Iterated on SFT output format across two training runs: identified that long evidence fields caused 14% JSON truncation rate; introduced compact chain-of-thought calculation steps (`subtract(206588, 181001)=25587; divide(25587, 181001)=14.1%`) and a 350-char evidence cap, eliminating truncation entirely (100% JSON validity) and reducing inference latency 2.6× (51.7 s → 19.6 s per question)

**Retrieval-Augmented Generation**
- Built a two-stage RAG pipeline — dense retrieval (BGE-large + FAISS IndexFlatIP) followed by cross-encoder reranking (MS-MARCO MiniLM) — and ran a controlled 5-system ablation isolating the contribution of each component to numerical accuracy (0% → 7% → 15.3%); error analysis identified retrieval failure as the binding constraint (71% of errors), with model conditional accuracy of ~60% when the correct chunk is retrieved
- Diagnosed and fixed two retrieval corpus bugs: (1) a training/inference distribution mismatch where prose chunks were indexed against a model trained on compressed table texts, dropping accuracy from 15.3% to 3%; (2) a corpus coverage gap where 118 of 177 evaluation source documents were missing from the retrieval index, causing near-zero accuracy on the affected questions
- Implemented and evaluated BM25+dense hybrid retrieval with RRF fusion; found it degraded accuracy (15.3% → 13.3%) due to 24% cross-company contamination — FinQA questions use identical templates across companies, so BM25 keyword matching finds the right metric from the wrong company; documented as a negative finding and reverted to pure dense retrieval

**GPU & Systems Debugging**
- Diagnosed fp16 attention overflow on NVIDIA Pascal (sm_61): K-values reaching 419 over 128-dim heads produce QK^T sums of ~3.7M, exceeding fp16 max (65,504); fixed by setting `torch_dtype=float32` and disabling AMP
- Resolved CUDA OOM on DPO training by enabling `precompute_ref_log_probs=True`, halving peak memory by caching reference logprobs before the training loop

**Evaluation & Data Engineering**
- Designed a deterministic eval harness scoring JSON schema validity, numerical accuracy (0.1% tolerance), and refusal accuracy across 5 ablation systems on 300 stratified held-out questions
- Extracted and resolved multi-step arithmetic programs from FinQA's structured annotation format (handling both single-qa and multi-qa records) to build 3,439 chain-of-thought training pairs from the full train+dev split; applied ticker-stratified splits to prevent company-level leakage

---

## Results

Evaluated on 300 held-out questions from FinQA, stratified by company ticker (no leakage). The retrieval corpus uses the same compressed table-format texts the SFT model was trained on to ensure a fair evaluation.

### Ablation: component contributions (v1 — accuracy-maximizing)

| System | JSON Valid | Num. Accuracy | Refusal Acc. | Avg Latency |
|---|---|---|---|---|
| `base_no_rag` | 97.3% | 0.0% | 14.7% | 8.7 s |
| `base_simple_rag` | 63.3% | 7.0% | 52.7% | 11.7 s |
| `base_two_stage_rag` | 68.0% | 6.7% | 53.7% | 11.6 s |
| `sft_two_stage_rag` (v1) | 97.3% | **15.3%** | 97.3% | 51.7 s |
| `sft_dpo_two_stage_rag` | — | — | — | — |

### SFT iteration: v1 vs v2b

| Configuration | Training examples | JSON Valid | Num. Accuracy | Refusal Acc. | Avg Latency |
|---|---|---|---|---|---|
| v1 — compact calc, 2109 ex | 2,109 | 97.3% | **15.3%** | 97.3% | 51.7 s |
| v2b — CoT calc, capped evidence, 3439 ex | 3,439 | **100%** | 13.0% | **100%** | **19.6 s** |

v2b trades 2.3 pp of numerical accuracy for perfect JSON validity, perfect refusal accuracy, and **2.6× faster inference** — a better engineering configuration for a production setting where reliability and latency matter more than marginal accuracy gains.

**Key findings:**

- **Retrieval adds 7 pp numerical accuracy** over the no-RAG baseline (0% → 7%), confirming that the base model lacks the financial figures in its parametric memory.
- **SFT adds another 8 pp** on top of retrieval (7% → 15.3%), a **2.2× lift** from learning to parse and calculate from the retrieved context.
- **Refusal accuracy jumps from 14.7% to 97.3%** after SFT — the model learned when to answer vs. decline, the primary signal in the SFT training data.
- **Dense vs. two-stage retrieval is a wash** (7.0% vs. 6.7%) — the cross-encoder reranker adds no measurable lift on this compact, keyword-dense corpus.
- **More data + CoT steps improve output reliability** — v2b (3,439 examples, compact chain-of-thought) achieves 100% JSON validity and 2.6× lower latency vs. v1 by capping the evidence field in training targets, preventing output truncation.
- **DPO collapsed** — the DPO adapter generates degenerate repetitive output (the ⚗ token loop). Root cause: synthetic rejected responses were too easily distinguishable (rule-perturbed arithmetic), causing the DPO objective to saturate at step 1 with loss ≈ 0 and near-zero gradients for 390 of 400 steps. The optimizer drifted the adapter weights into a degenerate attractor. This is a documented failure mode of DPO when `beta` is too low (0.1) and rejected samples are trivially separable; hard negatives generated by the policy model itself are needed.

---

## Architecture

```
FinQA data (train.json)
        │
        ▼
┌───────────────────┐
│  data_pipeline.py │  chunk, split by ticker (no cross-split leakage)
└───────────────────┘
        │
        ├── data/processed/chunks.jsonl    ← retrieval corpus
        ├── data/sft/train.jsonl           ← SFT training pairs
        └── data/dpo/train.jsonl           ← DPO chosen/rejected pairs

        │
        ▼
┌──────────────────────┐        ┌──────────────────────────┐
│  retrieval_engine.py │        │  alignment.py            │
│  FAISS IndexFlatIP   │        │  QLoRA SFT (1000 steps)  │
│  + cross-encoder     │        │  QLoRA DPO (400 steps)   │
└──────────────────────┘        └──────────────────────────┘
        │ top-5 chunks                    │ adapter weights
        └──────────────┬──────────────────┘
                       ▼
              ┌──────────────────────┐
              │  rag_pipeline.py     │
              │  retrieve → prompt   │
              │  → generate → JSON   │
              └──────────────────────┘
                       │ predictions.jsonl
                       ▼
              ┌──────────────────────┐
              │  eval_harness.py     │
              │  JSON validity       │
              │  numerical accuracy  │
              │  refusal accuracy    │
              └──────────────────────┘
```

---

## Hardware & Training Details

**Hardware:** Single NVIDIA Titan X (Pascal, sm_61, 12 GB VRAM)

**Why not multi-GPU ZeRO-3?** Pascal (sm_61) lacks fp16 tensor cores. DeepSpeed ZeRO-3 in fp32 requires 7B × 4 bytes ÷ 4 GPUs = 7 GB/GPU in sharded weights alone — plus cuBLAS workspace and NCCL buffers this exceeds 12 GB. ZeRO-3 with CPU offload triggered a 2.03 GiB CUDA OOM on the `embed_tokens` all-gather after `deepspeed.initialize()` consumed 6.4 GB of unexplained initialization overhead.

**Solution — QLoRA on a single GPU:**
- 4-bit NF4 quantization (bitsandbytes): 7B × 0.5 bytes ≈ **3.8 GB** for the frozen base
- LoRA rank-16 adapters on all attention + MLP projections: **~134 MB** trainable
- `torch_dtype=float32` for activations — Pascal fp16 overflows in QK^T attention at layer 27 (K-values reach 419, sum over 128 head-dim reaches 3.7M, exceeds fp16 max 65,504)
- `paged_adamw_8bit` optimizer, gradient checkpointing, batch size 1 × grad accum 16
- No DeepSpeed, no torchrun — plain `python -m src.alignment`

**SFT training (v1):** 1000 steps, cosine LR (2e-4 → 0), loss: 0.74 → 0.055, ~16 hours
**SFT training (v2b):** 1500 steps on 3,439 examples, loss: 1.753 → 0.079, ~24 hours

**DPO training:** 400 steps, β=0.1, `precompute_ref_log_probs=True` (halves GPU memory by caching reference logprobs upfront), ~13 hours — but collapsed (see Results above)

---

## Ablation Systems

| # | System | Retrieval | Model |
|---|--------|-----------|-------|
| 0 | `base_no_rag` | none | Qwen2.5-7B-Instruct |
| 1 | `base_simple_rag` | dense FAISS cosine | Qwen2.5-7B-Instruct |
| 2 | `base_two_stage_rag` | dense + cross-encoder rerank | Qwen2.5-7B-Instruct |
| 3 | `sft_two_stage_rag` | dense + cross-encoder rerank | + SFT LoRA adapter |
| 4 | `sft_dpo_two_stage_rag` | dense + cross-encoder rerank | + DPO LoRA adapter |

All systems use the same ChatML prompt, greedy decoding, and output schema — enabling apples-to-apples scoring.

---

## Installation

```bash
# GPU FAISS must come from conda (PyPI build lacks faiss.StandardGpuResources)
conda install -c pytorch -c nvidia faiss-gpu

# PyTorch with CUDA (match your driver — check nvidia-smi)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# All other dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### 1 — Data pipeline (v1)

```bash
python -m src.data_pipeline \
    --input     data/raw/documents.jsonl \
    --output-dir data/processed \
    --config    configs/default.yaml
```

Produces `data/processed/sft_chunks.jsonl` (retrieval corpus) and `data/sft/train.jsonl` (2,109 training pairs).

### 1b — Data pipeline (v2 — full FinQA + CoT)

```bash
python -m src.build_sft_data \
    --config configs/v2.yaml
```

Produces `data/processed/sft_chunks_v2.jsonl` (4,878 chunks, all val source docs covered) and `data/sft/train_v2.jsonl` (3,439 CoT training pairs).

### 2 — Generate DPO pairs (from SFT data)

```bash
python -m src.gen_dpo_data \
    --sft_data data/sft/train.jsonl \
    --out      data/dpo/train.jsonl
```

### 3 — SFT training (single GPU)

```bash
# Smoke run (5 steps)
CUDA_VISIBLE_DEVICES=0 python -m src.alignment \
    --mode sft --config configs/default.yaml \
    --data data/sft/train.jsonl --debug

# v1 — 1000 steps (~16 h on Titan X Pascal)
CUDA_VISIBLE_DEVICES=0 python -m src.alignment \
    --mode sft --config configs/default.yaml \
    --data data/sft/train.jsonl

# v2b — 1500 steps on full FinQA + CoT (~24 h)
CUDA_VISIBLE_DEVICES=0 python -m src.alignment \
    --mode sft --config configs/v2.yaml \
    --data data/sft/train_v2.jsonl
```

Adapters saved to `outputs/sft_adapter/` (v1) and `outputs/sft_adapter_v2b/` (v2b).

### 4 — DPO training (single GPU)

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.alignment \
    --mode dpo --config configs/default.yaml \
    --data data/dpo/train.jsonl \
    --sft_adapter outputs/sft_adapter
```

Adapter saved to `outputs/dpo_adapter/`.

### 5 — Inference (all 5 systems)

```bash
# v1 corpus + adapter
CUDA_VISIBLE_DEVICES=0 python run_inference.py
# or a subset:
CUDA_VISIBLE_DEVICES=0 python run_inference.py --systems sft_two_stage_rag

# v2b corpus + adapter
CUDA_VISIBLE_DEVICES=0 python run_inference.py --v2
```

Predictions written to `outputs/predictions/<system>.jsonl` (v1) or `outputs/predictions/v2/<system>.jsonl` (v2b).

### 6 — Evaluate

```bash
python -m src.eval_harness \
    --predictions outputs/predictions/sft_two_stage_rag.jsonl \
    --report      outputs/reports/sft_two_stage_rag.json
```

---

## Repository Layout

```
configs/
  default.yaml          hyperparameters for v1 (model, retrieval, training)
  v2.yaml               overrides for v2b (1500 steps, v2 adapter/corpus paths)
data/
  processed/            chunks.jsonl, sft_chunks.jsonl, sft_chunks_v2.jsonl, questions.jsonl
  sft/                  train.jsonl (v1, 2109 ex), train_v2.jsonl (v2b, 3439 ex), val.jsonl
  dpo/                  train.jsonl (chosen/rejected pairs)
outputs/
  sft_adapter/          v1 SFT LoRA weights (tracked via Git LFS)
  sft_adapter_v2b/      v2b SFT LoRA weights (compact CoT, capped evidence)
  dpo_adapter/          DPO LoRA weights (collapsed — see Results)
  predictions/          v1 per-system prediction JSONL files
  predictions/v2/       v2b per-system prediction JSONL files
  reports/              per-system eval JSON reports
src/
  data_pipeline.py      ingestion, chunking, ticker-stratified split (v1)
  build_sft_data.py     full FinQA CoT extraction + hybrid corpus builder (v2b)
  retrieval_engine.py   GPU FAISS + BGE-large embedder + cross-encoder reranker
  alignment.py          QLoRA SFT and DPO training (single-GPU, bitsandbytes)
  gen_dpo_data.py       synthetic DPO pair generation from SFT data
  rag_pipeline.py       end-to-end inference for all 5 ablation systems
  eval_harness.py       deterministic scoring (JSON schema, numerical match, refusal)
run_inference.py        convenience wrapper — runs all systems sequentially (--v2 flag for v2b)
```

---

## Known Limitations

- **Numerical accuracy ceiling (~15%)** — error analysis shows 71% of failures are retrieval failures (the correct number is not in the top-5 retrieved chunks). The model's conditional accuracy given correct retrieval is ~60%. Closing the retrieval gap is the binding constraint, not model quality. Hard negative mining and company-aware retrieval filtering are the most promising next steps.
- **BM25+dense hybrid retrieval does not help (negative finding)** — adding BM25 with RRF fusion reduced v1 accuracy from 15.3% to 13.3% and dropped JSON validity from 97.3% to 79%. Root cause: FinQA questions use identical templates across companies ("What was the change in net income from 2008 to 2009?"); BM25 keyword matching finds the right metric but from the wrong company, introducing 24% cross-company contamination in the top-1 chunk. Dense retrieval handles company disambiguation via semantic context. Hybrid retrieval is not viable without an explicit company-filtering stage.
- **DPO requires hard negatives** — rule-perturbed synthetic rejections (wrong arithmetic, swapped operators) are trivially distinguishable. The DPO objective saturates immediately, and extended optimization with a low beta collapses the adapter. Use policy-sampled rejected completions and β ≥ 0.3 for stable DPO on this domain.
- **Pascal (sm_61) fp16 overflow** — Qwen2.5-7B's attention at layer 27 produces K-values up to ~420 in fp16; the QK^T sum over a 128-dim head reaches ~3.7M, overflowing fp16 max (65,504). Loading with `torch_dtype=float32` avoids this but requires `fp16=False` in the Trainer (no AMP autocasting).
