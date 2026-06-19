"""Generate DPO chosen/rejected pairs from SFT training data.

Each SFT example has a correct target_json (→ chosen). This script produces a
plausible-but-wrong rejected response for every example, drawn randomly from
four failure modes that represent real LLM mistakes on financial QA:

  1. arithmetic_error   — right formula, wrong arithmetic result
  2. wrong_formula      — wrong operation (% when absolute asked, or vice versa)
  3. hallucinated_value — picks a different number from the context
  4. false_confidence   — refuses to flag insufficient context / low confidence

Output: data/dpo/train.jsonl with keys: text, question, chosen, rejected
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Number extraction helpers
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")


def _parse_num(s: str) -> float | None:
    s = s.replace(",", "").rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def _find_context_numbers(text: str) -> list[float]:
    """Extract all numeric values from the RAG context."""
    nums = []
    for m in _NUM_RE.finditer(text):
        v = _parse_num(m.group())
        if v is not None and abs(v) > 0:
            nums.append(v)
    return list(set(nums))


# ---------------------------------------------------------------------------
# Answer perturbation strategies
# ---------------------------------------------------------------------------

def _perturb_numeric(answer: str, rng: random.Random) -> str:
    """Shift the numeric value in the answer by ±10–35%."""
    v = _parse_num(answer.strip())
    if v is None or v == 0:
        return answer
    factor = rng.choice([
        1 + rng.uniform(0.10, 0.35),
        1 - rng.uniform(0.10, 0.35),
    ])
    new_v = v * factor
    if "%" in answer:
        return f"{new_v:.2f}%"
    if abs(new_v) >= 100:
        return f"{new_v:.1f}"
    return f"{new_v:.4f}"


def _corrupt_calculation(calc: str, rng: random.Random) -> str:
    """Swap one arithmetic operator in the calculation string."""
    swaps = [("+", "-"), ("-", "+"), ("*", "/"), ("/", "*")]
    rng.shuffle(swaps)
    for src, dst in swaps:
        # Only swap inside the operator symbols, not in variable names
        if src in calc:
            # Replace first occurrence
            return calc.replace(src, dst, 1)
    return calc


def _hallucinated_value(answer: str, context_nums: list[float], rng: random.Random) -> str:
    """Replace the answer with a plausible-looking value pulled from context."""
    candidates = [n for n in context_nums if _parse_num(answer) != n]
    if not candidates:
        return _perturb_numeric(answer, rng)
    chosen = rng.choice(candidates)
    if "%" in answer:
        return f"{chosen:.2f}%"
    if abs(chosen) >= 100:
        return f"{chosen:.1f}"
    return f"{chosen:.4f}"


# ---------------------------------------------------------------------------
# Rejection builders
# ---------------------------------------------------------------------------

def _arithmetic_error(tgt: dict, context_nums: list[float], rng: random.Random) -> dict:
    """Right structure, wrong number — classic arithmetic mistake."""
    new_answer = _perturb_numeric(tgt["answer"], rng)
    new_calc = tgt.get("calculation") or ""
    if new_calc:
        # Replace the final result in the calculation string
        final = tgt["answer"].replace(",", "").rstrip("%")
        new_calc = new_calc.replace(final, new_answer.rstrip("%"), 1)
    return {
        "answer": new_answer,
        "calculation": new_calc,
        "evidence": tgt.get("evidence", ""),
        "confidence": round(rng.uniform(0.80, 0.90), 2),
        "insufficient_context": False,
    }


def _wrong_formula(tgt: dict, context_nums: list[float], rng: random.Random) -> dict:
    """Swap operation: percentage ↔ absolute, or multiply ↔ divide."""
    calc = tgt.get("calculation") or ""
    new_calc = _corrupt_calculation(calc, rng)
    # Re-derive a plausible (but wrong) answer from context numbers
    if context_nums:
        new_answer = _perturb_numeric(tgt["answer"], rng)
    else:
        new_answer = tgt["answer"]
    return {
        "answer": new_answer,
        "calculation": new_calc,
        "evidence": tgt.get("evidence", ""),
        "confidence": round(rng.uniform(0.75, 0.88), 2),
        "insufficient_context": False,
    }


def _hallucinated_number(tgt: dict, context_nums: list[float], rng: random.Random) -> dict:
    """Pick a wrong number from the context — common grounding failure."""
    new_answer = _hallucinated_value(tgt["answer"], context_nums, rng)
    # Also corrupt the evidence slightly by picking a wrong sentence
    evidence = tgt.get("evidence", "")
    sentences = [s.strip() for s in evidence.split(";") if s.strip()]
    if len(sentences) > 1:
        # Drop or reorder a sentence to break grounding
        rng.shuffle(sentences)
        new_evidence = "; ".join(sentences[:-1])
    else:
        new_evidence = evidence
    return {
        "answer": new_answer,
        "calculation": tgt.get("calculation", ""),
        "evidence": new_evidence,
        "confidence": round(rng.uniform(0.70, 0.85), 2),
        "insufficient_context": False,
    }


def _false_confidence(tgt: dict, context_nums: list[float], rng: random.Random) -> dict:
    """Model claims high confidence / not-insufficient when it should hedge."""
    # Either: refuse to flag insufficient context, or give low confidence on real answer
    mode = rng.choice(["overconfident", "wrong_flag"])
    if mode == "overconfident":
        # Correct answer but absurdly high confidence with no evidence
        return {
            "answer": tgt["answer"],
            "calculation": "",
            "evidence": "",
            "confidence": 0.99,
            "insufficient_context": False,
        }
    else:
        # Sets insufficient_context=True when we actually have the answer
        return {
            "answer": "",
            "calculation": "",
            "evidence": tgt.get("evidence", ""),
            "confidence": 0.0,
            "insufficient_context": True,
        }


# ---------------------------------------------------------------------------
# Main pair-generation logic
# ---------------------------------------------------------------------------

_STRATEGIES = [
    _arithmetic_error,
    _wrong_formula,
    _hallucinated_number,
    _false_confidence,
]


def make_rejected(tgt: dict, context: str, rng: random.Random) -> dict:
    context_nums = _find_context_numbers(context)
    strategy = rng.choice(_STRATEGIES)
    return strategy(tgt, context_nums, rng)


def generate_dpo_pairs(
    sft_path: str,
    out_path: str,
    seed: int = 42,
    max_examples: int | None = None,
) -> int:
    rng = random.Random(seed)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    strategy_counts: dict[str, int] = {s.__name__: 0 for s in _STRATEGIES}

    with open(sft_path) as fin, open(out_path, "w") as fout:
        rows = [json.loads(l) for l in fin if l.strip()]
        if max_examples:
            rows = rows[:max_examples]

        for row in rows:
            context = row.get("text") or row.get("context", "")
            question = row["question"]
            chosen_str = row["target_json"]

            try:
                tgt = json.loads(chosen_str)
            except json.JSONDecodeError:
                continue

            context_nums = _find_context_numbers(context)
            strategy = rng.choice(_STRATEGIES)
            strategy_counts[strategy.__name__] = strategy_counts.get(strategy.__name__, 0) + 1
            rejected = strategy(tgt, context_nums, rng)
            rejected_str = json.dumps(rejected, ensure_ascii=False)

            fout.write(json.dumps({
                "text": context,
                "question": question,
                "chosen": chosen_str,
                "rejected": rejected_str,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} DPO pairs → {out_path}")
    print("Strategy distribution:")
    for name, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count} ({100*count/written:.1f}%)")
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DPO chosen/rejected pairs.")
    parser.add_argument("--sft_data", default="data/sft/train.jsonl")
    parser.add_argument("--out", default="data/dpo/train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()
    generate_dpo_pairs(args.sft_data, args.out, args.seed, args.max_examples)


if __name__ == "__main__":
    main()
