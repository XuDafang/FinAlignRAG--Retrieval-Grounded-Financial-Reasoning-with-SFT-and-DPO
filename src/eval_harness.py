"""FinAlignRAG — Step 3: Evaluation Harness (Robust Financial Verification).

Deterministic scoring for the ablation study. Every public scorer is defensive:
malformed input returns ``False`` rather than raising.

Metrics
-------
* JSON schema validity        — does ``predicted_json`` parse and have all keys?
* Numerical match             — predicted vs gold answer within 0.1% rel. tol,
                                after normalizing financial formats ($4.2B,
                                4.2 billion, 4,200M, 4,200,000,000, 5.2%).
* Evidence support            — do the *source operands* of the calculation
                                appear in the evidence (derived values need not).
* Refusal accuracy            — does ``insufficient_context`` match the label?
* Retrieval recall@k          — when gold chunk ids are provided.
* Latency                     — mean ``latency_ms``.

Public API
----------
``evaluate_json_validity(predicted_json) -> bool``
``evaluate_numerical_match(predicted_json, ground_truth_answer, tolerance=0.001) -> bool``
``evaluate_evidence_support(evidence_text, math_calculation) -> bool``
``evaluate_refusal_accuracy(predicted_json, should_refuse) -> bool``
``evaluate_prediction_file(predictions_path, output_report_path) -> dict``

CLI
---
``python -m src.eval_harness --predictions preds.jsonl --report reports/metrics.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger("finalignrag.eval_harness")

# Required keys for the answer JSON schema (matches the SFT/prediction schema).
REQUIRED_JSON_KEYS: tuple[str, ...] = (
    "answer",
    "calculation",
    "evidence",
    "confidence",
    "insufficient_context",
)

# Retrieval recall is reported @k.
RECALL_K = 5

# Relative tolerance used when matching operands against evidence numbers.
_EVIDENCE_TOL = 0.001

# Magnitude suffixes -> multiplier (finance: M == million, MM == million).
_SUFFIX: dict[str, float] = {
    "t": 1e12, "trillion": 1e12, "trillions": 1e12,
    "b": 1e9, "bn": 1e9, "billion": 1e9, "billions": 1e9,
    "m": 1e6, "mm": 1e6, "million": 1e6, "millions": 1e6,
    "k": 1e3, "thousand": 1e3, "thousands": 1e3,
}

# Pure scaling constants that may appear in a calculation (e.g. "* 100" for a
# percentage) but are NOT document-sourced operands, so evidence need not contain them.
_SCALING_CONSTANTS: set[float] = {100.0, 1000.0}

# Candidate numeric token: optional $, sign, digits w/ commas, decimals, an
# optional magnitude suffix (single letters guarded by a no-letter lookahead so
# "5 to" does not parse "t" as trillion), and an optional trailing %.
_NUM_RE = re.compile(
    r"\$?[-+]?\d[\d,]*(?:\.\d+)?\s*"
    r"(?:trillions?|billions?|millions?|thousands?|bn|mm|[tbmk])?(?![a-z])"
    r"\s*%?",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Numeric normalization helpers
# ---------------------------------------------------------------------------
def _parse_number(token: str) -> tuple[float, float] | None:
    """Parse one numeric token -> ``(full_value, mantissa)`` or ``None``.

    ``full_value`` applies the magnitude suffix (4.2B -> 4.2e9); ``mantissa`` is
    the bare number before scaling (4.2B -> 4.2). Keeping both lets evidence
    matching succeed whether the calculation writes "96.9" and the evidence
    writes "96.9B" (mantissa match) or both write the fully expanded number.
    """
    s = token.strip().lower()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").strip().lstrip("$").strip()
    if s.endswith("%"):
        s = s[:-1].strip()
    s = s.replace(",", "").strip()

    m = re.fullmatch(r"([+-]?\d*\.?\d+)\s*([a-z]+)?", s)
    if not m:
        return None
    try:
        mantissa = float(m.group(1))
    except ValueError:
        return None

    scale = 1.0
    suffix = m.group(2)
    if suffix:
        if suffix.endswith("s") and suffix[:-1] in _SUFFIX:
            suffix = suffix[:-1]
        if suffix not in _SUFFIX:
            return None
        scale = _SUFFIX[suffix]

    full = mantissa * scale
    if neg:
        full, mantissa = -full, -mantissa
    return full, mantissa


def _extract_numbers(text: Any) -> list[tuple[float, float]]:
    """Extract all ``(full_value, mantissa)`` pairs from free text."""
    if not text:
        return []
    out: list[tuple[float, float]] = []
    for match in _NUM_RE.finditer(str(text)):
        parsed = _parse_number(match.group(0))
        if parsed is not None:
            out.append(parsed)
    return out


def _close(a: float, b: float, tol: float) -> bool:
    """True if ``a`` and ``b`` agree within relative tolerance ``tol``."""
    diff = abs(a - b)
    if diff == 0.0:
        return True
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return diff <= tol
    return diff / scale <= tol


def _load_json(predicted_json: Any) -> dict | None:
    """Safely parse a JSON string into a dict; return None on any failure."""
    try:
        data = json.loads(predicted_json)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


# ---------------------------------------------------------------------------
# Public scorers
# ---------------------------------------------------------------------------
def evaluate_json_validity(predicted_json: str) -> bool:
    """Parse ``predicted_json`` and verify all required schema keys are present."""
    data = _load_json(predicted_json)
    if data is None:
        return False
    return all(key in data for key in REQUIRED_JSON_KEYS)


def evaluate_numerical_match(
    predicted_json: str,
    ground_truth_answer: str,
    tolerance: float = 0.001,
) -> bool:
    """Compare predicted vs gold answer numbers within ``tolerance`` (relative).

    Financial formats are normalized first, so ``$4.2B``, ``4.2 billion``,
    ``4,200M`` and ``4,200,000,000`` all compare equal. Returns ``True`` when
    every gold number is matched by some predicted number.
    """
    data = _load_json(predicted_json)
    if data is None:
        return False
    predicted = [full for full, _ in _extract_numbers(data.get("answer", ""))]
    gold = [full for full, _ in _extract_numbers(ground_truth_answer)]
    if not predicted or not gold:
        return False
    return all(any(_close(g, p, tolerance) for p in predicted) for g in gold)


def evaluate_evidence_support(evidence_text: str, math_calculation: str) -> bool:
    """Verify the calculation's *source operands* appear in the evidence text.

    Derived values (percentages, ratios, differences) are NOT required — only
    the operands that should have been read from the document. Pure scaling
    constants (100, 1000) are ignored. Matching is scale-insensitive: an operand
    ``96.9`` is supported by evidence containing ``96.9B``.
    """
    operands = [
        (full, mant)
        for full, mant in _extract_numbers(math_calculation)
        if full not in _SCALING_CONSTANTS
    ]
    if not operands:
        return False

    evidence = _extract_numbers(evidence_text)
    if not evidence:
        return False
    evidence_values: list[float] = []
    for full, mant in evidence:
        evidence_values.extend((full, mant))

    for full, mant in operands:
        operand_values = {full, mant}
        if not any(
            _close(ov, ev, _EVIDENCE_TOL)
            for ov in operand_values
            for ev in evidence_values
        ):
            return False
    return True


def evaluate_refusal_accuracy(predicted_json: str, should_refuse: bool) -> bool:
    """Check whether ``insufficient_context`` matches the expected refusal label."""
    data = _load_json(predicted_json)
    if data is None:
        return False
    ic = _coerce_bool(data.get("insufficient_context"))
    if ic is None:
        return False
    return ic == bool(should_refuse)


# ---------------------------------------------------------------------------
# Aggregate evaluation over a prediction file
# ---------------------------------------------------------------------------
def _recall_at_k(record: dict, k: int = RECALL_K) -> float | None:
    """Retrieval recall@k for one record, or None if no gold chunk ids present."""
    gold = record.get("relevant_chunk_ids") or record.get("gold_chunk_ids")
    if not gold:
        return None
    gold_set = set(gold)
    if not gold_set:
        return None
    retrieved = [
        c.get("chunk_id") for c in record.get("retrieved_chunks", [])[:k]
    ]
    return len(gold_set & set(retrieved)) / len(gold_set)


def evaluate_prediction_file(predictions_path: str, output_report_path: str) -> dict:
    """Score a JSONL prediction file and write aggregate metrics as JSON.

    Each line must follow the Prediction JSONL schema (``predicted_json``,
    ``ground_truth_answer``, ``should_refuse``, ``retrieved_chunks``,
    ``latency_ms``; optional ``relevant_chunk_ids`` enables recall@k).
    """
    records: list[dict] = []
    with open(predictions_path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL record on line %d", lineno)

    json_valid: list[bool] = []
    numerical: list[bool] = []
    evidence: list[bool] = []
    refusal: list[bool] = []
    recalls: list[float] = []
    latencies: list[float] = []

    for record in records:
        pj = record.get("predicted_json", "")
        valid = evaluate_json_validity(pj)
        json_valid.append(valid)

        should_refuse = bool(record.get("should_refuse", False))
        refusal.append(evaluate_refusal_accuracy(pj, should_refuse))

        # Numerical / evidence metrics only apply to answerable (non-refusal) items.
        if not should_refuse:
            numerical.append(
                evaluate_numerical_match(pj, record.get("ground_truth_answer", ""))
            )
            if valid:
                data = _load_json(pj) or {}
                calculation = data.get("calculation", "")
                if calculation:
                    evidence.append(
                        evaluate_evidence_support(data.get("evidence", ""), calculation)
                    )

        recall = _recall_at_k(record)
        if recall is not None:
            recalls.append(recall)

        latency = record.get("latency_ms")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            latencies.append(float(latency))

    def rate(values: list) -> float | None:
        return (sum(values) / len(values)) if values else None

    metrics = {
        "num_samples": len(records),
        "json_validity_rate": rate(json_valid),
        "numerical_accuracy": rate(numerical),
        "evidence_support_accuracy": rate(evidence),
        "refusal_accuracy": rate(refusal),
        f"retrieval_recall_at_{RECALL_K}": rate(recalls),
        "avg_latency_ms": rate(latencies),
        "counts": {
            "json_validity": len(json_valid),
            "numerical": len(numerical),
            "evidence_support": len(evidence),
            "refusal": len(refusal),
            "retrieval_recall": len(recalls),
            "latency": len(latencies),
        },
    }

    parent = os.path.dirname(output_report_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Wrote evaluation report (%d samples) to %s", len(records), output_report_path)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _format_summary(metrics: dict) -> str:
    def pct(x: float | None) -> str:
        return "N/A" if x is None else f"{x * 100:.2f}%"

    lat = metrics.get("avg_latency_ms")
    return (
        f"Samples            : {metrics['num_samples']}\n"
        f"JSON validity       : {pct(metrics['json_validity_rate'])}\n"
        f"Numerical accuracy  : {pct(metrics['numerical_accuracy'])}\n"
        f"Evidence support    : {pct(metrics['evidence_support_accuracy'])}\n"
        f"Refusal accuracy    : {pct(metrics['refusal_accuracy'])}\n"
        f"Retrieval recall@{RECALL_K} : {pct(metrics[f'retrieval_recall_at_{RECALL_K}'])}\n"
        f"Avg latency (ms)    : {'N/A' if lat is None else f'{lat:.1f}'}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval_harness",
        description="Score a JSONL prediction file and write an evaluation report.",
    )
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL.")
    parser.add_argument("--report", required=True, help="Output path for the metrics report (JSON).")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    metrics = evaluate_prediction_file(args.predictions, args.report)
    print(_format_summary(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
