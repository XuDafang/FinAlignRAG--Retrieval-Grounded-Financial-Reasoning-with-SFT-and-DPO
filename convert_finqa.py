import json, pathlib

out = pathlib.Path("data/raw/documents.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

total = 0
with out.open("w") as fh:
    for split in ["train.json", "dev.json"]:
        path = pathlib.Path(f"data/{split}")
        records = json.loads(path.read_text())
        for r in records:
            ticker = r["filename"].split("/")[0]
            doc_id = r["filename"].replace("/", "_").replace(".pdf", "")
            text   = " ".join(r.get("pre_text", []) + r.get("post_text", []))
            if text.strip():
                fh.write(json.dumps({"ticker": ticker, "source_doc_id": doc_id, "text": text}) + "\n")
                total += 1
        print(f"  {split}: {len(records)} records")

print(f"Total written: {total} -> {out}")
