"""Build the SFT dataset, retrieval corpus, and evaluation questions.

The FinQA training split supplies supervised examples. The development split
stays held out and supplies evaluation questions. Gold evidence from both
splits is included in the retrieval corpus so development questions have
retrievable context without leaking their answers into SFT training.

FinQA calculation steps are converted to a compact trace such as::

    subtract(206588, 181001)=25587; divide(25587, 181001)=14.1%

``#N`` back-references are resolved to intermediate results so every operation
contains concrete values.

Outputs:
  data/sft/train.jsonl
  data/processed/sft_chunks.jsonl
  data/processed/questions.jsonl

Usage:
  python -m src.build_sft_data
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator
from typing import Any


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

_EVIDENCE_MAX_CHARS = 350


def _op_name(op: str) -> str:
    """Map a FinQA op string such as ``minus2-1`` to ``subtract``."""
    key = re.sub(r"[\d\-]+$", "", op.lower())
    return _OP_MAP.get(key, key or op)


def build_calculation_trace(steps: list[dict[str, Any]]) -> str:
    """Build a compact calculation trace and resolve intermediate references."""
    results: list[str] = []
    parts: list[str] = []

    for step in steps:
        op = _op_name(step.get("op", ""))
        arg1 = str(step.get("arg1", ""))
        arg2 = str(step.get("arg2", ""))
        result = str(step.get("res", ""))

        def resolve(arg: str) -> str:
            match = re.match(r"^#(\d+)$", arg)
            if match:
                index = int(match.group(1))
                return results[index] if index < len(results) else arg
            return arg

        resolved_arg1 = resolve(arg1)
        resolved_arg2 = resolve(arg2)
        results.append(result)

        if resolved_arg2:
            parts.append(f"{op}({resolved_arg1}, {resolved_arg2})={result}")
        else:
            parts.append(f"{op}({resolved_arg1})={result}")

    return "; ".join(parts)


def _cap_evidence(text: str, max_chars: int = _EVIDENCE_MAX_CHARS) -> str:
    """Truncate evidence at a statement boundary when possible."""
    if len(text) <= max_chars:
        return text
    cut = text.rfind(";", 0, max_chars)
    if cut > max_chars // 2:
        return text[: cut + 1].strip()
    return text[:max_chars].strip()


def _filename_to_ids(filename: str) -> tuple[str, str]:
    """Convert a FinQA filename into ticker and source-document identifiers."""
    source_doc_id = filename.replace(".pdf", "").replace("/", "_")
    ticker = filename.split("/")[0]
    return ticker, source_doc_id


def _get_qa_entries(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Read both single-QA and multi-QA FinQA records."""
    if "qa" in example:
        return [example["qa"]]
    return [
        example[key]
        for key in sorted(example)
        if key.startswith("qa_")
    ]


def _load_records(paths: list[str]) -> Iterator[dict[str, Any]]:
    """Yield records from one or more FinQA JSON files."""
    for path in paths:
        with open(path, encoding="utf-8") as file:
            yield from json.load(file)


def _make_sft_record(
    example: dict[str, Any],
    qa: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert one FinQA question into a supervised training record."""
    question = qa.get("question", "").strip()
    steps = qa.get("steps") or []
    gold_evidence = qa.get("gold_inds") or {}
    answer = str(qa.get("answer") or qa.get("exe_ans", "")).strip()
    if not question or not steps or not gold_evidence or not answer:
        return None

    context = " ".join(gold_evidence.values()).strip()
    if not context:
        return None

    ticker, source_doc_id = _filename_to_ids(example["filename"])
    target = {
        "answer": answer,
        "calculation": build_calculation_trace(steps),
        "evidence": _cap_evidence(context),
        "confidence": 0.95,
        "insufficient_context": False,
    }
    return {
        "ticker": ticker,
        "source_doc_id": source_doc_id,
        "text": context,
        "question": question,
        "target_json": json.dumps(target, ensure_ascii=False),
    }


def _add_chunk(
    chunks: dict[tuple[str, str], dict[str, Any]],
    chunk_counter: defaultdict[str, int],
    ticker: str,
    source_doc_id: str,
    text: str,
) -> str:
    """Add a unique evidence chunk and return its stable identifier."""
    key = (source_doc_id, text)
    if key not in chunks:
        index = chunk_counter[source_doc_id]
        chunk_counter[source_doc_id] += 1
        chunks[key] = {
            "chunk_id": f"{source_doc_id}_{index:03d}",
            "text": text,
            "ticker": ticker,
            "source_doc_id": source_doc_id,
            "chunk_index": index,
        }
    return chunks[key]["chunk_id"]


def _write_jsonl(path: str, records: Iterable[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build(
    train_files: list[str],
    eval_files: list[str],
    out_sft: str,
    out_chunks: str,
    out_questions: str,
) -> None:
    """Build all generated data required by SFT training and evaluation."""
    sft_records: list[dict[str, Any]] = []
    evaluation_questions: list[dict[str, Any]] = []
    chunks: dict[tuple[str, str], dict[str, Any]] = {}
    chunk_counter: defaultdict[str, int] = defaultdict(int)
    seen_training_examples: set[tuple[str, str]] = set()
    seen_evaluation_examples: set[tuple[str, str]] = set()

    for example in _load_records(train_files):
        ticker, source_doc_id = _filename_to_ids(example["filename"])
        for qa in _get_qa_entries(example):
            for text in (qa.get("gold_inds") or {}).values():
                if text:
                    _add_chunk(chunks, chunk_counter, ticker, source_doc_id, text)

            record = _make_sft_record(example, qa)
            if record is None:
                continue

            key = (source_doc_id, record["question"].lower())
            if key not in seen_training_examples:
                seen_training_examples.add(key)
                sft_records.append(record)

    for example in _load_records(eval_files):
        ticker, source_doc_id = _filename_to_ids(example["filename"])
        for qa_index, qa in enumerate(_get_qa_entries(example)):
            question = qa.get("question", "").strip()
            answer = str(qa.get("answer") or qa.get("exe_ans", "")).strip()
            if not question or not answer:
                continue

            key = (source_doc_id, question.lower())
            if key in seen_evaluation_examples:
                continue
            seen_evaluation_examples.add(key)

            relevant_chunk_ids: list[str] = []
            for text in (qa.get("gold_inds") or {}).values():
                if not text:
                    continue
                chunk_id = _add_chunk(
                    chunks, chunk_counter, ticker, source_doc_id, text
                )
                if chunk_id not in relevant_chunk_ids:
                    relevant_chunk_ids.append(chunk_id)

            evaluation_questions.append({
                "id": f"{source_doc_id}_q{qa_index:02d}",
                "ticker": ticker,
                "source_doc_id": source_doc_id,
                "question": question,
                "ground_truth_answer": answer,
                "should_refuse": False,
                "relevant_chunk_ids": relevant_chunk_ids,
            })

    _write_jsonl(out_sft, sft_records)
    _write_jsonl(out_chunks, chunks.values())
    _write_jsonl(out_questions, evaluation_questions)

    print(f"SFT training pairs  : {len(sft_records):,} -> {out_sft}")
    print(f"Retrieval chunks    : {len(chunks):,} -> {out_chunks}")
    print(f"Evaluation questions: {len(evaluation_questions):,} -> {out_questions}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.build_sft_data",
        description="Build SFT rows, retrieval chunks, and held-out questions.",
    )
    parser.add_argument(
        "--train_files",
        nargs="+",
        default=["data/train.json"],
        help="FinQA files used for SFT training (default: data/train.json).",
    )
    parser.add_argument(
        "--eval_files",
        nargs="+",
        default=["data/dev.json"],
        help="Held-out FinQA files used for evaluation (default: data/dev.json).",
    )
    parser.add_argument("--out_sft", default="data/sft/train.jsonl")
    parser.add_argument("--out_chunks", default="data/processed/sft_chunks.jsonl")
    parser.add_argument("--out_questions", default="data/processed/questions.jsonl")
    args = parser.parse_args(argv)
    build(
        args.train_files,
        args.eval_files,
        args.out_sft,
        args.out_chunks,
        args.out_questions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
