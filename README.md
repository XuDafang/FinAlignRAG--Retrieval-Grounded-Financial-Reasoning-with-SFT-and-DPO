# FinAlignRAG: Learning SFT and DPO with Financial Question Answering

FinAlignRAG is a small, end-to-end project for learning two model-alignment
stages:

- **Supervised fine-tuning (SFT):** teach a language model to answer financial
  questions in a grounded JSON format.
- **Direct Preference Optimization (DPO):** continue from the SFT adapter and
  teach the model to prefer a correct response over a plausible incorrect one.

The project uses FinQA financial-report questions and a retrieval-augmented
generation (RAG) pipeline. It is an educational alignment experiment, not a
general system for reading arbitrary PDFs or performing complete financial
analysis.

## What the Project Does

Consider one example from FinQA:

```text
Retrieved context:
Net cash from operating activities was 206,588 in 2009 and 181,001 in 2008.

Question:
What was the percentage change from 2008 to 2009?

Calculation:
(206588 - 181001) / 181001 = 14.1%
```

The model should return a grounded, machine-readable response:

```json
{
  "answer": "14.1%",
  "calculation": "subtract(206588, 181001)=25587; divide(25587, 181001)=14.1%",
  "evidence": "Net cash from operating activities was 206,588 in 2009 and 181,001 in 2008.",
  "confidence": 0.95,
  "insufficient_context": false
}
```

Each part of the project has a different job:

- **RAG supplies the facts.** It retrieves the relevant table or text evidence
  containing the financial figures.
- **SFT teaches the task.** It shows the model thousands of
  `context + question -> correct JSON response` examples so the model learns
  which numbers to use, how to calculate the answer, and how to follow the
  output schema.
- **DPO teaches preferences.** It presents a correct response alongside a
  plausible but incorrect response and trains the model to prefer the correct
  one.

SFT is the core training stage. DPO is included as an optional learning
experiment; it is not automatically better, especially when rejected responses
are synthetic or too easy to distinguish. A practical version of this system
can reasonably stop after SFT.

There is one supported data path, one configuration, and one set of output
locations.

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

## Models

The project uses three pretrained models:

| Full model name | Hugging Face ID | Role | Trained here? | Why it was chosen |
|---|---|---|---|---|
| **Alibaba Qwen 2.5 7B Instruct** | `Qwen/Qwen2.5-7B-Instruct` | Generates the grounded financial answer | Yes: SFT, followed optionally by DPO | Open weights permit local alignment training; instruction tuning provides a useful starting point; the 7B size can be adapted on a 12 GB GPU with QLoRA |
| **Beijing Academy of Artificial Intelligence General Embedding Large English, Version 1.5** | `BAAI/bge-large-en-v1.5` | Embeds questions and evidence for dense FAISS retrieval | No: used frozen | Provides strong general-purpose English semantic retrieval and works naturally with normalized-vector similarity search |
| **MS MARCO (Microsoft Machine Reading Comprehension) Cross-Encoder based on MiniLM-L6, Version 2** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranks the retrieved evidence candidates | No: used frozen | Scores each question-evidence pair jointly while remaining smaller and faster than a large reranking model |

Only Qwen is adapted by this project. QLoRA keeps its base weights in 4-bit form
and trains small rank-16 LoRA adapters. SFT teaches the task and output format;
DPO optionally adjusts preferences between correct and plausible incorrect
answers. The embedding and reranking models are not jointly trained with Qwen.

These models are practical choices for the learning objective and available
hardware, not the winners of a comprehensive model-selection benchmark. The
`v1.5` and `v2` suffixes are part of the external model names and are
unrelated to project versions.

## How to Read the Code

Follow the data through the project instead of reading files alphabetically.
This order introduces the SFT/DPO idea first and leaves GPU infrastructure
until later:

1. **Read this README and the configuration.**
   [`configs/default.yaml`](configs/default.yaml) provides a compact view of
   the selected models, retrieval settings, LoRA parameters, SFT steps, and DPO
   settings.

2. **Read the smallest behavioral examples.**
   [`tests/test_build_sft_data.py`](tests/test_build_sft_data.py) shows how a
   FinQA record becomes an SFT row, retrieval chunks, and held-out questions.
   [`tests/test_eval_harness.py`](tests/test_eval_harness.py) shows what the
   project considers a correct model response.

3. **Follow raw FinQA data into training data.**
   In [`src/build_sft_data.py`](src/build_sft_data.py), begin with
   `_make_sft_record()` and then read `build()`. Together they produce:

   ```text
   data/train.json -> SFT examples
   data/dev.json   -> held-out evaluation questions
   both            -> retrieval evidence chunks
   ```

4. **Study SFT and DPO prompt formatting and training.**
   [`src/alignment.py`](src/alignment.py) is the central learning file. Read
   `_build_prompt()`, `format_sft_samples()`, `format_dpo_pairs()`,
   `run_sft()`, and `run_dpo()` in that order.

5. **See how DPO preferences are constructed.**
   [`src/gen_dpo_data.py`](src/gen_dpo_data.py) creates synthetic rejected
   responses with arithmetic, formula, grounding, and confidence errors. Read
   the four rejection strategies before `generate_dpo_pairs()`.

6. **Learn the retrieval layer.**
   [`src/retrieval_engine.py`](src/retrieval_engine.py) defines
   `FinancialRetrievalEngine`, which embeds evidence, searches the FAISS
   index, and reranks candidates. The embedding and reranking models remain
   frozen.

7. **Connect retrieval to answer generation.**
   [`src/rag_pipeline.py`](src/rag_pipeline.py) defines `RAGPipeline`:

   ```text
   question -> retrieve evidence -> build prompt -> generate JSON
   ```

8. **Read orchestration and evaluation last.**
   [`run_inference.py`](run_inference.py) runs the selected base, SFT, and DPO
   systems. [`src/eval_harness.py`](src/eval_harness.py) measures JSON
   validity, numerical accuracy, evidence support, refusal behavior, retrieval
   recall, and latency.

For a short introduction focused only on model alignment, use:

```text
README -> build_sft_data.py -> alignment.py -> gen_dpo_data.py -> tests
```

Leave `retrieval_engine.py` and `run_inference.py` until later; they contain
more GPU and orchestration details than the core SFT/DPO workflow.

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

These two JSON files are kept local and ignored by Git because they are large.

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
