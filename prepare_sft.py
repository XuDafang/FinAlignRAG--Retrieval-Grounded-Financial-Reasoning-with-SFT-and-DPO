"""Convert FinQA train/dev splits into SFT training JSONL.

Each output record:
  ticker, source_doc_id, text (gold context), question, target_json

target_json contains: answer, calculation, evidence, confidence,
insufficient_context — the schema alignment.py trains the model to produce.
"""

import json
import pathlib


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def table_to_text(table: list) -> str:
    """Serialize a FinQA table (list of rows) to a readable text block."""
    return "\n".join(" | ".join(str(cell) for cell in row) for row in table)


def steps_to_calc(steps: list) -> str:
    """Convert FinQA step list to a readable math expression string.

    Example steps:
      [{"op": "minus2-1", "arg1": "206588", "arg2": "181001", "res": "25587"},
       {"op": "divide2-2", "arg1": "#0", "arg2": "181001", "res": "14.1%"}]
    → "(206588 - 181001) = 25587; (#0 / 181001) = 14.1%"
    """
    _OP = {
        "minus": "-", "subtract": "-",
        "add": "+", "plus": "+",
        "multiply": "*", "times": "*",
        "divide": "/",
    }
    intermediate: dict[str, str] = {}
    parts: list[str] = []

    for i, step in enumerate(steps):
        op = step["op"].split("2-")[0] if "2-" in step["op"] else step["op"]
        a1 = intermediate.get(step["arg1"], step["arg1"]) if step["arg1"].startswith("#") else step["arg1"]
        a2_raw = step.get("arg2", "")
        a2 = intermediate.get(a2_raw, a2_raw) if a2_raw.startswith("#") else a2_raw
        res = step.get("res", "")

        sym = _OP.get(op)
        expr = f"({a1} {sym} {a2})" if sym else f"{op}({a1}, {a2})"
        intermediate[f"#{i}"] = expr
        parts.append(f"{expr} = {res}")

    return "; ".join(parts)


def build_context(record: dict) -> str:
    """Concatenate pre_text, table, and post_text into a single context block."""
    parts = []
    if record.get("pre_text"):
        parts.append(" ".join(record["pre_text"]))
    if record.get("table"):
        parts.append(table_to_text(record["table"]))
    if record.get("post_text"):
        parts.append(" ".join(record["post_text"]))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def convert(input_path: str, output_path: str) -> int:
    records = json.loads(pathlib.Path(input_path).read_text())
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with out.open("w") as fh:
        for r in records:
            qa = r.get("qa", {})
            question = qa.get("question", "").strip()
            answer = str(qa.get("answer") or qa.get("exe_ans", "")).strip()
            if not question or not answer:
                skipped += 1
                continue

            ticker = r["filename"].split("/")[0]
            doc_id = r["filename"].replace("/", "_").replace(".pdf", "")

            # Gold evidence: the specific passages the answer derives from.
            # Used as context (simulates what an ideal retrieval system returns)
            # rather than the full document — keeps sequences short and teaches
            # the model to reason from retrieved snippets, not full filings.
            gold_inds = qa.get("gold_inds", {})
            evidence = " ".join(gold_inds.values()) if gold_inds else ""
            context = evidence if evidence else build_context(r)[:1000]

            # Human-readable calculation from step list; fall back to program string
            steps = qa.get("steps", [])
            calculation = steps_to_calc(steps) if steps else qa.get("program", "")

            target_json = json.dumps({
                "answer": answer,
                "calculation": calculation,
                "evidence": evidence,
                "confidence": 0.95,
                "insufficient_context": False,
            }, ensure_ascii=False)

            fh.write(json.dumps({
                "ticker": ticker,
                "source_doc_id": doc_id,
                "text": context,
                "question": question,
                "target_json": target_json,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"  {input_path}: {written} written, {skipped} skipped → {output_path}")
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Preparing SFT training data...")
    n_train = convert("data/train.json", "data/sft/train.jsonl")
    n_val   = convert("data/dev.json",   "data/sft/val.jsonl")
    print(f"Done — {n_train + n_val} total SFT samples.")

    # Spot-check first record
    first = json.loads(open("data/sft/train.jsonl").readline())
    print("\n--- Spot check (first record) ---")
    print("ticker      :", first["ticker"])
    print("question    :", first["question"])
    tj = json.loads(first["target_json"])
    print("answer      :", tj["answer"])
    print("calculation :", tj["calculation"])
    print("evidence    :", tj["evidence"][:120], "...")
