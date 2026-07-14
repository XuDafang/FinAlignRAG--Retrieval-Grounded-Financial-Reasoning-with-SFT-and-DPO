# FinAlignRAG: Learning SFT and DPO with Financial Question Answering

FinAlignRAG is a small, end-to-end project for learning two model-alignment
stages:

- **Supervised fine-tuning (SFT):** teach a language model to answer financial
  questions in a grounded JSON format.
- **Direct Preference Optimization (DPO):** continue from the SFT adapter and
  teach the model to prefer a correct response over a plausible incorrect one.

The project uses FinQA financial-report questions and a retrieval-augmented
generation (RAG) pipeline. There is one supported data path, one configuration,
and one set of output locations.

## Pipeline

```text
data/train.json
      |
      |  python -m src.build_sft_data
      v
data/sft/train.jsonl ------------------------> SFT training
      |                                          |
      |  python -m src.gen_dpo_data              v
      v                                  outputs/sft_adapter/
data/dpo/train.jsonl                            |
      |                                          | initialize from SFT
      +------------------------------------------+
                                                 v
                                         DPO training
                                                 |
                                                 v
                                         outputs/dpo_adapter/

data/dev.json
      |
      +--> data/processed/sft_chunks.jsonl --> retrieval index
      +--> data/processed/questions.jsonl  --> inference and evaluation
```

The base model is Qwen2.5-7B-Instruct. Training uses QLoRA: the base model is
loaded in 4-bit form while small LoRA adapter weights are trained.

## Dataset

FinQA is already extracted into JSON. This repository does **not** download or
parse PDF files. A value such as `JKHY/2009/page_28.pdf` in the `filename`
field identifies the source filing page; the model reads the extracted text,
tables, questions, and annotations stored in JSON.

### Source files

| File | Contents | Use in this project |
|---|---|---|
| `data/train.json` | 3,037 filing-page records and 3,965 QA annotations | Source for SFT examples and training-side retrieval evidence |
| `data/dev.json` | 421 filing-page records and 542 QA annotations | Held-out source for evaluation questions and retrieval evidence |
| `data/test_private.json` | 434 private-test records without public gold QA labels | Not used by the default learning pipeline |
| `data/train_turn.json` | 11,104 conversational training turns | Optional turn-level experiments; not used by default |
| `data/dev_turn.json` | 1,490 conversational development turns | Optional turn-level experiments; not used by default |
| `data/test_turn_private.json` | 1,521 private conversational test turns | Not used by the default learning pipeline |

The JSON files are kept local and ignored by Git because they are large.

### Record structure

A typical FinQA record has this shape:

```json
{
  "filename": "JKHY/2009/page_28.pdf",
  "pre_text": ["Narrative text before the table."],
  "table": [
    ["", "2009", "2008"],
    ["net income", "$ 103102", "$ 104222"]
  ],
  "table_ori": [
    ["", "2009", "2008"],
    ["Net income", "$103,102", "$104,222"]
  ],
  "post_text": ["Narrative text after the table."],
  "qa": {
    "question": "What was the percentage change ...?",
    "answer": "14.1%",
    "program": "subtract(206588, 181001), divide(#0, 181001)",
    "steps": [
      {"op": "minus2-1", "arg1": "206588", "arg2": "181001", "res": "25587"},
      {"op": "divide2-2", "arg1": "#0", "arg2": "181001", "res": "14.1%"}
    ],
    "gold_inds": {
      "table_6": "net cash from operating activities was 206588 in 2009 and 181001 in 2008"
    }
  }
}
```

Some records use `qa_0`, `qa_1`, and similar keys instead of a single
`qa` key. The data builder supports both forms.

### Table storage

Tables are not stored as images. Both `table` and `table_ori` are nested
arrays:

- The outer array is the sequence of rows.
- Each row is an array of cell strings.
- `table` contains normalized text.
- `table_ori` is closer to the original formatting.

The training builder normally uses the flattened table or text evidence in
`qa.gold_inds`. This keeps prompts compact and directly ties each answer to
the evidence used by FinQA annotators.

## Generated Data

Run:

```bash
python -m src.build_sft_data
```

The command uses `train.json` for SFT and keeps `dev.json` held out. In the
included dataset it creates 3,321 SFT rows, 4,878 unique evidence chunks, and
458 deduplicated evaluation questions.

| File | Structure | Used by |
|---|---|---|
| `data/sft/train.jsonl` | `ticker`, `source_doc_id`, `text`, `question`, `target_json` | SFT training and DPO-pair generation |
| `data/processed/sft_chunks.jsonl` | `chunk_id`, `text`, `ticker`, `source_doc_id`, `chunk_index` | FAISS retrieval during inference |
| `data/processed/questions.jsonl` | `id`, `question`, `ground_truth_answer`, `should_refuse`, `relevant_chunk_ids` | Held-out inference and evaluation |
| `data/dpo/train.jsonl` | `text`, `question`, `chosen`, `rejected` | DPO training |

Generated data, adapters, indexes, predictions, logs, and reports are ignored
by Git. Their directories contain `.gitkeep` files so the expected layout is
visible in a fresh clone.

## SFT Data

Each SFT row contains a context and question plus the exact response the model
should learn to produce:

```json
{
  "text": "net cash from operating activities was ...",
  "question": "What was the percentage change ...?",
  "target_json": "{\"answer\": \"14.1%\", \"calculation\": \"subtract(...)=...; divide(...)=14.1%\", \"evidence\": \"...\", \"confidence\": 0.95, \"insufficient_context\": false}"
}
```

`src/alignment.py` turns this into a ChatML prompt followed by the target JSON
completion. SFT teaches response structure, calculation-trace formatting, and
grounding behavior.

## DPO Data

`src/gen_dpo_data.py` converts every SFT target into a preference pair:

```json
{
  "text": "retrieved financial evidence",
  "question": "the financial question",
  "chosen": "the correct SFT target JSON",
  "rejected": "a plausible but incorrect JSON response"
}
```

Rejected responses simulate arithmetic errors, incorrect formulas,
hallucinated values, and confidence mistakes. This is intentionally a simple
learning baseline. Synthetic negatives can become too easy for the model, so
inspect the pairs and monitor DPO loss and reward margins during experiments.
Policy-generated hard negatives are a useful next step.

## Installation

The training and retrieval paths require an NVIDIA GPU. The original hardware
target was a 12 GB Pascal GPU, so the code uses eager attention and float32
activations around the 4-bit model.

```bash
conda create -n finalignrag python=3.11
conda activate finalignrag

# Install a CUDA build of PyTorch that matches the local driver.
pip install torch --index-url https://download.pytorch.org/whl/cu121

# GPU FAISS is normally installed through conda on Linux or WSL2.
conda install -c pytorch faiss-gpu

pip install -r requirements.txt
```

The tests and data builder run without model training. GPU inference requires a
FAISS build exposing `faiss.StandardGpuResources`.

## Training

### 1. Prepare data

```bash
python -m src.build_sft_data
```

### 2. Inspect an SFT row

```bash
python -c "import json; print(json.dumps(json.loads(open('data/sft/train.jsonl', encoding='utf-8').readline()), indent=2))"
```

### 3. Train the SFT adapter

Start with a five-step smoke run:

```bash
python -m src.alignment \
  --mode sft \
  --config configs/default.yaml \
  --data data/sft/train.jsonl \
  --debug
```

Remove `--debug` for the configured 1,500-step run. The adapter is written to
`outputs/sft_adapter/`.

### 4. Generate DPO preference pairs

```bash
python -m src.gen_dpo_data
```

Inspect several `chosen` and `rejected` pairs before training. A preference
dataset is useful only when the rejected response is wrong for a meaningful
reason and is not distinguishable by a trivial formatting cue.

### 5. Train the DPO adapter

```bash
python -m src.alignment \
  --mode dpo \
  --config configs/default.yaml \
  --data data/dpo/train.jsonl \
  --sft_adapter outputs/sft_adapter \
  --debug
```

DPO must start from the trained SFT adapter. Remove `--debug` for the
configured 400-step run. The resulting adapter is written to
`outputs/dpo_adapter/`.

## Inference and Evaluation

Run the SFT and DPO systems:

```bash
python run_inference.py --systems \
  sft_two_stage_rag \
  sft_dpo_two_stage_rag
```

Predictions are written to `outputs/predictions/<system>.jsonl`. The first RAG
run builds and saves `outputs/sft_faiss_index/`; later runs reuse it.

Evaluate each prediction file:

```bash
python -m src.eval_harness \
  --predictions outputs/predictions/sft_two_stage_rag.jsonl \
  --report reports/sft_metrics.json

python -m src.eval_harness \
  --predictions outputs/predictions/sft_dpo_two_stage_rag.jsonl \
  --report reports/dpo_metrics.json
```

The report includes JSON validity, numerical accuracy, evidence support,
refusal accuracy, retrieval recall, and average latency.

## Repository Layout

```text
configs/
  default.yaml             model, retrieval, SFT, and DPO settings
data/
  train.json               local FinQA training split
  dev.json                 local held-out development split
  *_turn.json              optional conversational variants
  processed/               generated chunks and evaluation questions
  sft/                     generated supervised examples
  dpo/                     generated preference pairs
outputs/
  sft_adapter/             generated SFT LoRA adapter
  dpo_adapter/             generated DPO LoRA adapter
  predictions/             generated inference results
reports/                   generated evaluation reports
src/
  build_sft_data.py        FinQA extraction and split preparation
  alignment.py             QLoRA SFT and DPO training
  gen_dpo_data.py          preference-pair generation
  retrieval_engine.py      dense retrieval and reranking
  rag_pipeline.py          prompting, generation, and prediction writing
  eval_harness.py          evaluation metrics
tests/
  test_build_sft_data.py
  test_eval_harness.py
run_inference.py           runs selected comparison systems
```

## Learning Notes

- SFT and DPO solve different problems. SFT first teaches the desired task and
  output distribution; DPO then changes relative preferences within that
  distribution.
- DPO does not replace SFT. Its policy must be initialized from the SFT adapter
  used to construct the preference task.
- The default DPO negatives are synthetic. Their quality is the largest
  limitation of the preference-learning experiment.
- The default evaluation questions are answerable FinQA questions, so refusal
  accuracy is not a balanced refusal benchmark. Add explicit unanswerable
  examples before drawing conclusions about refusal behavior.
- The retrieval corpus uses FinQA gold evidence. This isolates alignment
  behavior, but it is easier than retrieving from complete SEC filings.
