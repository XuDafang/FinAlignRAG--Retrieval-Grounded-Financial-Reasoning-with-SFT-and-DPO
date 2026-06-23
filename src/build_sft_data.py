"""Build CoT-enriched SFT training data from the full FinQA dataset.

Reads data/train.json + data/dev.json (3458 raw examples, including multi-question
records with qa_0/qa_1 keys), produces ~4100 training pairs after:
  - deduplicating by question
  - excluding held-out val questions (data/processed/questions.jsonl)
  - dropping examples without program steps or gold evidence

Chain-of-thought improvement over v1:
  v1 calculation:  "(206588 - 181001) = 25587; ((206588 - 181001) / 181001) = 14.1%"
  v2 calculation:  "Step 1: subtract(206588, 181001) = 25587; Step 2: divide(25587, 181001) = 14.1%"

Steps are taken directly from qa.steps and #N back-references are resolved to
the actual intermediate result so the model sees concrete numbers at every step.

Outputs:
  data/sft/train_v2.jsonl          SFT training pairs
  data/processed/sft_chunks_v2.jsonl   retrieval corpus for new FAISS index

Usage:
  python -m src.build_sft_data
  python -m src.build_sft_data --out_sft data/sft/train_v2.jsonl \
      --out_chunks data/processed/sft_chunks_v2.jsonl \
      --val_questions data/processed/questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Any


# ---------------------------------------------------------------------------
# FinQA op-code → human-readable name
# ---------------------------------------------------------------------------
_OP_MAP = {
    "minus": "subtract",
    "subtract": "subtract",
    "add": "add",
    "divide": "divide",
    "multiply": "multiply",
    "greater": "greater_than",
    "exp": "exp",
    "table_max": "table_max",
    "table_min": "table_min",
    "table_sum": "table_sum",
    "table_average": "table_average",
}


def _op_name(op: str) -> str:
    """Map a FinQA op string like 'minus2-1' → 'subtract'."""
    key = re.sub(r"[\d\-]+$", "", op.lower())
    return _OP_MAP.get(key, key or op)


def build_cot(steps: list[dict[str, Any]]) -> str:
    """Build a semicolon-separated CoT string from FinQA qa.steps.

    Back-references (#N) are resolved to the actual intermediate result so the
    model sees concrete numbers at every step rather than opaque placeholders.
    """
    results: list[str] = []
    parts: list[str] = []

    for i, step in enumerate(steps):
        op = _op_name(step.get("op", ""))
        arg1 = str(step.get("arg1", ""))
        arg2 = str(step.get("arg2", ""))
        res = str(step.get("res", ""))

        def resolve(arg: str) -> str:
            m = re.match(r"^#(\d+)$", arg)
            if m:
                idx = int(m.group(1))
                return results[idx] if idx < len(results) else arg
            return arg

        a1 = resolve(arg1)
        a2 = resolve(arg2)
        results.append(res)

        if a2:
            parts.append(f"Step {i + 1}: {op}({a1}, {a2}) = {res}")
        else:
            parts.append(f"Step {i + 1}: {op}({a1}) = {res}")

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# FinQA record helpers
# ---------------------------------------------------------------------------
def _filename_to_ids(filename: str) -> tuple[str, str]:
    """'JKHY/2009/page_28.pdf' → ('JKHY', 'JKHY_2009_page_28')."""
    base = filename.replace(".pdf", "").replace("/", "_")
    ticker = filename.split("/")[0]
    return ticker, base


def _get_qa_entries(ex: dict[str, Any]) -> list[dict[str, Any]]:
    """Handle both single-qa {'qa': {...}} and multi-qa {'qa_0': {...}, 'qa_1': {...}}."""
    if "qa" in ex:
        return [ex["qa"]]
    return [ex[k] for k in sorted(ex.keys()) if k.startswith("qa_")]


def _process_qa(
    ex: dict[str, Any],
    qa: dict[str, Any],
    val_questions: set[str],
) -> dict[str, Any] | None:
    """Convert one FinQA (example, qa) pair to an SFT record, or None to skip."""
    question = qa.get("question", "").strip()
    if not question or question.lower() in val_questions:
        return None

    steps = qa.get("steps") or []
    if not steps:
        return None

    gold_inds = qa.get("gold_inds") or {}
    if not gold_inds:
        return None

    text = " ".join(gold_inds.values()).strip()
    if not text:
        return None

    ticker, source_doc_id = _filename_to_ids(ex["filename"])
    cot = build_cot(steps)
    answer = qa.get("answer", "").strip()

    target = {
        "answer": answer,
        "calculation": cot,
        "evidence": text,
        "confidence": 0.95,
        "insufficient_context": False,
    }

    return {
        "ticker": ticker,
        "source_doc_id": source_doc_id,
        "text": text,
        "question": question,
        "target_json": json.dumps(target, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build(
    finqa_files: list[str],
    val_questions_path: str,
    out_sft: str,
    out_chunks: str,
) -> None:
    val_questions: set[str] = set()
    with open(val_questions_path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            val_questions.add(d["question"].strip().lower())

    sft_records: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    # chunk corpus: keyed by (source_doc_id, text) for deduplication
    chunks: dict[tuple[str, str], dict[str, Any]] = {}
    chunk_counter: defaultdict[str, int] = defaultdict(int)

    for fname in finqa_files:
        with open(fname, encoding="utf-8") as fh:
            data = json.load(fh)

        for ex in data:
            for qa in _get_qa_entries(ex):
                record = _process_qa(ex, qa, val_questions)
                if record is None:
                    continue

                q_key = record["question"].lower()
                if q_key in seen_questions:
                    continue
                seen_questions.add(q_key)

                sft_records.append(record)

                # Accumulate retrieval corpus
                ck = (record["source_doc_id"], record["text"])
                if ck not in chunks:
                    sid = record["source_doc_id"]
                    idx = chunk_counter[sid]
                    chunk_counter[sid] += 1
                    chunks[ck] = {
                        "chunk_id": f"{sid}_{idx:03d}",
                        "text": record["text"],
                        "ticker": record["ticker"],
                        "source_doc_id": sid,
                        "chunk_index": idx,
                    }

    os.makedirs(os.path.dirname(out_sft) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_chunks) or ".", exist_ok=True)

    with open(out_sft, "w", encoding="utf-8") as fh:
        for rec in sft_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(out_chunks, "w", encoding="utf-8") as fh:
        for chunk in chunks.values():
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"SFT training pairs : {len(sft_records):,}  →  {out_sft}")
    print(f"Retrieval corpus   : {len(chunks):,} chunks  →  {out_chunks}")
    print(f"Val questions excluded: {len(val_questions)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.build_sft_data",
        description="Build CoT-enriched SFT data from full FinQA dataset.",
    )
    parser.add_argument(
        "--finqa_files",
        nargs="+",
        default=["data/train.json", "data/dev.json"],
        help="FinQA JSON files to process (default: train + dev).",
    )
    parser.add_argument("--out_sft", default="data/sft/train_v2.jsonl")
    parser.add_argument("--out_chunks", default="data/processed/sft_chunks_v2.jsonl")
    parser.add_argument(
        "--val_questions",
        default="data/processed/questions.jsonl",
        help="JSONL of held-out eval questions to exclude from training.",
    )
    args = parser.parse_args(argv)
    build(args.finqa_files, args.val_questions, args.out_sft, args.out_chunks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
