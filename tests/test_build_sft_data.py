import json

from src.build_sft_data import build, build_calculation_trace


def _write_json(path, records):
    path.write_text(json.dumps(records), encoding="utf-8")


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_calculation_trace_resolves_intermediate_results():
    steps = [
        {"op": "minus2-1", "arg1": "206588", "arg2": "181001", "res": "25587"},
        {"op": "divide2-2", "arg1": "#0", "arg2": "181001", "res": "14.1%"},
    ]

    assert build_calculation_trace(steps) == (
        "subtract(206588, 181001)=25587; divide(25587, 181001)=14.1%"
    )


def test_build_keeps_eval_questions_out_of_sft(tmp_path):
    train_path = tmp_path / "train.json"
    eval_path = tmp_path / "dev.json"
    out_sft = tmp_path / "sft.jsonl"
    out_chunks = tmp_path / "chunks.jsonl"
    out_questions = tmp_path / "questions.jsonl"

    _write_json(train_path, [{
        "filename": "AAA/2020/page_1.pdf",
        "qa": {
            "question": "How much did revenue change?",
            "answer": "20",
            "steps": [
                {"op": "minus2-1", "arg1": "120", "arg2": "100", "res": "20"}
            ],
            "gold_inds": {"table_1": "Revenue was 120 in 2020 and 100 in 2019."},
        },
    }])
    _write_json(eval_path, [{
        "filename": "BBB/2021/page_2.pdf",
        "qa_0": {
            "question": "What was the margin?",
            "answer": "25%",
            "gold_inds": {"table_1": "Profit was 25 and revenue was 100."},
        },
        "qa_1": {
            "question": "What was profit?",
            "answer": "25",
            "gold_inds": {"table_1": "Profit was 25 and revenue was 100."},
        },
    }])

    build(
        [str(train_path)],
        [str(eval_path)],
        str(out_sft),
        str(out_chunks),
        str(out_questions),
    )

    sft_rows = _read_jsonl(out_sft)
    chunks = _read_jsonl(out_chunks)
    questions = _read_jsonl(out_questions)

    assert [row["question"] for row in sft_rows] == ["How much did revenue change?"]
    assert {chunk["ticker"] for chunk in chunks} == {"AAA", "BBB"}
    assert len(chunks) == 2
    assert [row["question"] for row in questions] == [
        "What was the margin?",
        "What was profit?",
    ]
    assert questions[0]["relevant_chunk_ids"] == questions[1]["relevant_chunk_ids"]
    assert all(row["should_refuse"] is False for row in questions)
