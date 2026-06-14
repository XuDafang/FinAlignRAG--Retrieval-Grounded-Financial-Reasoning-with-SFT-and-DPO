"""FinAlignRAG — Step 1: Data Pipeline (Ingestion & Leakage Control).

Responsibilities
----------------
1. Ingestion      : load raw financial documents from a JSONL manifest.
2. Chunking       : split each document into overlapping, metadata-tagged chunks.
3. Split-by-ticker: partition chunks into train/val/test by company ticker.
4. Leakage control: guarantee no ticker appears in more than one split.

Public API
----------
``chunk_document(text, ticker, source_doc_id, chunk_size=512, overlap=50)``
    Returns a list of chunk dicts, each with EXACTLY the keys:
    ``chunk_id``, ``text``, ``ticker``, ``source_doc_id``, ``chunk_index``.

CLI
---
Run the full pipeline directly::

    python -m src.data_pipeline \
        --input data/raw/documents.jsonl \
        --output-dir data/processed \
        --chunk-size 512 --overlap 50 \
        --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15 --seed 42

Optionally load defaults from the centralized config::

    python -m src.data_pipeline --config configs/default.yaml \
        --input data/raw/documents.jsonl --output-dir data/processed

Notes
-----
``chunk_size`` and ``overlap`` are measured in *whitespace tokens*. This keeps
``chunk_document`` pure, deterministic, and dependency-free. A model-tokenizer
based chunker can be layered in later for exact alignment with the embedder's
512-token window without changing this public signature.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Any

logger = logging.getLogger("finalignrag.data_pipeline")

# Exact, ordered set of keys every chunk must expose (enforced in tests/CLI).
CHUNK_KEYS: tuple[str, ...] = (
    "chunk_id",
    "text",
    "ticker",
    "source_doc_id",
    "chunk_index",
)

# Required fields for each raw input document record.
DOCUMENT_KEYS: tuple[str, ...] = ("ticker", "source_doc_id", "text")


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------
def chunk_document(
    text: str,
    ticker: str,
    source_doc_id: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    """Split ``text`` into overlapping chunks tagged with document metadata.

    Each returned chunk is a dict with EXACTLY these keys: ``chunk_id``,
    ``text``, ``ticker``, ``source_doc_id``, ``chunk_index``.

    Parameters
    ----------
    text:
        Raw document text to chunk.
    ticker:
        Company ticker (preserved verbatim on every chunk).
    source_doc_id:
        Source document identifier (preserved verbatim on every chunk).
    chunk_size:
        Maximum number of whitespace tokens per chunk (default 512).
    overlap:
        Number of whitespace tokens shared between consecutive chunks
        (default 50). Must satisfy ``0 <= overlap < chunk_size``.

    Returns
    -------
    list[dict[str, Any]]
        Chunks in document order. ``chunk_index`` starts at 0 and increases
        monotonically within this document. ``chunk_id`` is formatted as
        ``f"{source_doc_id}_{chunk_index:03d}"`` (e.g. ``AAPL_2023_10K_000``).
        Returns an empty list when ``text`` contains no tokens.

    Raises
    ------
    ValueError
        If ``chunk_size <= 0`` or ``overlap`` is outside ``[0, chunk_size)``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < chunk_size; "
            f"got overlap={overlap}, chunk_size={chunk_size}"
        )

    tokens = text.split()
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks: list[dict[str, Any]] = []
    chunk_index = 0
    start = 0
    n = len(tokens)

    while start < n:
        window = tokens[start : start + chunk_size]
        chunks.append(
            {
                "chunk_id": f"{source_doc_id}_{chunk_index:03d}",
                "text": " ".join(window),
                "ticker": ticker,
                "source_doc_id": source_doc_id,
                "chunk_index": chunk_index,
            }
        )
        chunk_index += 1
        # Stop once this window reaches the end to avoid a redundant tail chunk.
        if start + chunk_size >= n:
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def load_documents(input_path: str) -> list[dict[str, Any]]:
    """Load raw documents from a JSONL file.

    Each non-empty line must be a JSON object containing at least the keys
    ``ticker``, ``source_doc_id`` and ``text``.

    Raises
    ------
    FileNotFoundError
        If ``input_path`` does not exist.
    ValueError
        If a line is not valid JSON or is missing required keys.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    documents: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON on line {lineno}: {exc}") from exc
            missing = [k for k in DOCUMENT_KEYS if k not in record]
            if missing:
                raise ValueError(
                    f"Document on line {lineno} missing required keys: {missing}"
                )
            documents.append(record)

    logger.info("Loaded %d documents from %s", len(documents), input_path)
    return documents


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    """Apply :func:`chunk_document` across a list of raw document records."""
    all_chunks: list[dict[str, Any]] = []
    for doc in documents:
        all_chunks.extend(
            chunk_document(
                text=doc["text"],
                ticker=doc["ticker"],
                source_doc_id=doc["source_doc_id"],
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )
    logger.info(
        "Produced %d chunks from %d documents (chunk_size=%d, overlap=%d)",
        len(all_chunks),
        len(documents),
        chunk_size,
        overlap,
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Split by ticker (leakage prevention)
# ---------------------------------------------------------------------------
def split_by_ticker(
    chunks: list[dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Partition chunks into train/val/test splits by company ticker.

    Whole tickers are assigned to a single split so that no company appears in
    more than one partition — the core leakage-prevention guarantee for RAG
    evaluation. Assignment is deterministic given ``seed``.

    Raises
    ------
    ValueError
        If the ratios do not sum to (approximately) 1.0.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got {total:.6f}"
        )

    tickers = sorted({c["ticker"] for c in chunks})
    rng = random.Random(seed)
    rng.shuffle(tickers)

    n = len(tickers)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    # Remainder goes to test to avoid dropping/duplicating tickers via rounding.
    train_tickers = set(tickers[:n_train])
    val_tickers = set(tickers[n_train : n_train + n_val])
    test_tickers = set(tickers[n_train + n_val :])

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for chunk in chunks:
        ticker = chunk["ticker"]
        if ticker in train_tickers:
            splits["train"].append(chunk)
        elif ticker in val_tickers:
            splits["val"].append(chunk)
        else:
            splits["test"].append(chunk)

    for name, split_tickers in (
        ("train", train_tickers),
        ("val", val_tickers),
        ("test", test_tickers),
    ):
        if not split_tickers:
            logger.warning(
                "Split '%s' has no tickers (only %d unique tickers available)",
                name,
                n,
            )
        logger.info(
            "Split '%s': %d tickers, %d chunks",
            name,
            len(split_tickers),
            len(splits[name]),
        )

    return splits


def assert_no_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Verify that ticker sets across splits are mutually disjoint.

    Raises
    ------
    AssertionError
        If any ticker appears in more than one split.
    """
    ticker_sets = {
        name: {c["ticker"] for c in chunks} for name, chunks in splits.items()
    }
    names = list(ticker_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = ticker_sets[names[i]] & ticker_sets[names[j]]
            assert not overlap, (
                f"Ticker leakage between '{names[i]}' and '{names[j]}': "
                f"{sorted(overlap)}"
            )
    logger.info("Leakage check passed: ticker sets are mutually disjoint.")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_jsonl(records: list[dict[str, Any]], path: str) -> None:
    """Write ``records`` to ``path`` as JSONL, creating parent dirs as needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def run_pipeline(
    input_path: str,
    output_dir: str,
    chunk_size: int = 512,
    overlap: int = 50,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """End-to-end Step 1 pipeline: ingest -> chunk -> split -> verify -> write.

    Writes ``chunks.jsonl`` plus ``train.jsonl`` / ``val.jsonl`` / ``test.jsonl``
    under ``output_dir`` and returns the in-memory splits.
    """
    documents = load_documents(input_path)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    splits = split_by_ticker(
        chunks,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    assert_no_leakage(splits)

    os.makedirs(output_dir, exist_ok=True)
    write_jsonl(chunks, os.path.join(output_dir, "chunks.jsonl"))
    for name, split_chunks in splits.items():
        write_jsonl(split_chunks, os.path.join(output_dir, f"{name}.jsonl"))

    return splits


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _load_config_defaults(config_path: str) -> dict[str, Any]:
    """Read chunking/split defaults from configs/default.yaml (best effort)."""
    import yaml  # local import: only needed when --config is supplied

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    data_cfg = cfg.get("data", {}) or {}
    split_cfg = data_cfg.get("split", {}) or {}
    return {
        "chunk_size": data_cfg.get("chunk_size"),
        "overlap": data_cfg.get("overlap"),
        "train_ratio": split_cfg.get("train_ratio"),
        "val_ratio": split_cfg.get("val_ratio"),
        "test_ratio": split_cfg.get("test_ratio"),
        "seed": cfg.get("project", {}).get("seed"),
        "output_dir": data_cfg.get("processed_dir"),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.data_pipeline",
        description=(
            "FinAlignRAG Step 1: ingest financial documents, chunk them, and "
            "split by ticker with leakage prevention."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a JSONL manifest of documents "
        "(each line: {ticker, source_doc_id, text}).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write chunks.jsonl and {train,val,test}.jsonl "
        "(default: data/processed, or data.processed_dir from --config).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to configs/default.yaml to source defaults from.",
    )
    parser.add_argument("--chunk-size", type=int, default=None, help="Tokens per chunk.")
    parser.add_argument("--overlap", type=int, default=None, help="Token overlap.")
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Hard-coded defaults; optionally overridden by --config; then by explicit flags.
    defaults: dict[str, Any] = {
        "chunk_size": 512,
        "overlap": 50,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "seed": 42,
        "output_dir": "data/processed",
    }
    if args.config:
        for key, value in _load_config_defaults(args.config).items():
            if value is not None:
                defaults[key] = value

    def resolve(flag_value: Any, key: str) -> Any:
        return flag_value if flag_value is not None else defaults[key]

    run_pipeline(
        input_path=args.input,
        output_dir=resolve(args.output_dir, "output_dir"),
        chunk_size=resolve(args.chunk_size, "chunk_size"),
        overlap=resolve(args.overlap, "overlap"),
        train_ratio=resolve(args.train_ratio, "train_ratio"),
        val_ratio=resolve(args.val_ratio, "val_ratio"),
        test_ratio=resolve(args.test_ratio, "test_ratio"),
        seed=resolve(args.seed, "seed"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
