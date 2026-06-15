"""Pytest suite for src/eval_harness.py.

Covers the cases mandated by the playbook: malformed JSON, percentage answers,
scaled financial numbers, missing evidence, and refusal labels (plus an
end-to-end prediction-file run).
"""

import json

from src import eval_harness as eh

# A complete, valid prediction matching the answer schema.
VALID = {
    "answer": "5.2%",
    "calculation": "(96.9 - 92.1) / 92.1",
    "evidence": "Net income was 96.9B vs 92.1B in the period.",
    "confidence": 0.92,
    "insufficient_context": False,
}


def _pj(d: dict) -> str:
    """Serialize an answer dict to the predicted_json string form."""
    return json.dumps(d)


# --- Malformed JSON: scorers must return False, never raise -----------------
def test_malformed_json_returns_false():
    assert eh.evaluate_json_validity("{ not valid json") is False
    assert eh.evaluate_json_validity("null") is False          # parses but not a dict
    assert eh.evaluate_json_validity("[1, 2, 3]") is False     # not a dict
    assert eh.evaluate_numerical_match("{bad", "5.2%") is False
    assert eh.evaluate_refusal_accuracy("{bad", True) is False


def test_json_validity_requires_all_keys():
    assert eh.evaluate_json_validity(_pj(VALID)) is True
    incomplete = dict(VALID)
    del incomplete["evidence"]
    assert eh.evaluate_json_validity(_pj(incomplete)) is False


# --- Percentage answers -----------------------------------------------------
def test_percentage_answers():
    assert eh.evaluate_numerical_match(_pj(VALID), "5.2%") is True
    assert eh.evaluate_numerical_match(_pj(VALID), "5.3%") is False  # > 0.1% rel diff
    # within 0.1% relative tolerance
    assert eh.evaluate_numerical_match(_pj(dict(VALID, answer="5.200%")), "5.2%") is True


# --- Scaled financial numbers: $4.2B == 4.2 billion == 4,200M == 4.2e9 ------
def test_scaled_financial_numbers():
    d = dict(VALID, answer="$4.2B")
    assert eh.evaluate_numerical_match(_pj(d), "4,200,000,000") is True
    assert eh.evaluate_numerical_match(_pj(d), "4.2 billion") is True
    assert eh.evaluate_numerical_match(_pj(d), "4,200M") is True
    assert eh.evaluate_numerical_match(_pj(d), "4.3 billion") is False


# --- Evidence support: source operands, not derived values ------------------
def test_evidence_support_operands_present():
    assert eh.evaluate_evidence_support(
        "Net income was 96.9B vs 92.1B.", "(96.9 - 92.1) / 92.1"
    ) is True


def test_evidence_support_missing_returns_false():
    assert eh.evaluate_evidence_support(
        "Revenue grew strongly across all segments.", "(96.9 - 92.1) / 92.1"
    ) is False


def test_evidence_support_ignores_derived_value():
    # The derived 5.2% is absent from evidence, but both operands are present.
    assert eh.evaluate_evidence_support(
        "Reported figures were 96.9 and 92.1 for the two years.",
        "(96.9 - 92.1) / 92.1",
    ) is True


def test_evidence_support_ignores_scaling_constant():
    # "* 100" is a scaling constant, not a document operand.
    assert eh.evaluate_evidence_support(
        "Figures: 96.9 and 92.1.", "(96.9 - 92.1) / 92.1 * 100"
    ) is True


# --- Refusal labels ---------------------------------------------------------
def test_refusal_labels():
    refusing = dict(VALID, insufficient_context=True)
    assert eh.evaluate_refusal_accuracy(_pj(refusing), True) is True
    assert eh.evaluate_refusal_accuracy(_pj(refusing), False) is False
    assert eh.evaluate_refusal_accuracy(_pj(VALID), False) is True
    assert eh.evaluate_refusal_accuracy(_pj(VALID), True) is False


# --- End-to-end prediction file ---------------------------------------------
def test_evaluate_prediction_file(tmp_path):
    rows = [
        {
            "id": "s1",
            "ground_truth_answer": "5.2%",
            "predicted_json": _pj(VALID),
            "should_refuse": False,
            "retrieved_chunks": [{"chunk_id": "AAPL_2023_10K_000"}],
            "relevant_chunk_ids": ["AAPL_2023_10K_000"],
            "latency_ms": 820,
        },
        {
            "id": "s2",
            "ground_truth_answer": "",
            "predicted_json": _pj(dict(VALID, insufficient_context=True)),
            "should_refuse": True,
            "retrieved_chunks": [],
            "latency_ms": 300,
        },
    ]
    preds = tmp_path / "preds.jsonl"
    preds.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    report = tmp_path / "metrics.json"

    metrics = eh.evaluate_prediction_file(str(preds), str(report))

    assert metrics["num_samples"] == 2
    assert metrics["json_validity_rate"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0
    assert metrics["numerical_accuracy"] == 1.0          # only s1 is answerable
    assert metrics["evidence_support_accuracy"] == 1.0
    assert metrics[f"retrieval_recall_at_{eh.RECALL_K}"] == 1.0
    assert metrics["avg_latency_ms"] == 560.0            # (820 + 300) / 2
    assert report.exists()


def test_malformed_jsonl_lines_are_skipped(tmp_path):
    preds = tmp_path / "preds.jsonl"
    preds.write_text(
        json.dumps({"predicted_json": _pj(VALID), "ground_truth_answer": "5.2%",
                    "should_refuse": False}) + "\n"
        + "{ this is not valid json\n"
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "metrics.json"
    metrics = eh.evaluate_prediction_file(str(preds), str(report))
    assert metrics["num_samples"] == 1  # malformed + blank lines skipped
