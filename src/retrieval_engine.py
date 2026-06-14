"""FinAlignRAG — Step 2: Retrieval Engine (Dense Retrieval + Cross-Encoder Rerank).

Two-stage RAG retrieval:
  1. Dense retrieval with FAISS ``IndexFlatIP`` over L2-normalized embeddings
     (inner product on unit vectors == cosine similarity).
  2. Cross-encoder reranking of the dense candidates.

GPU-ONLY POLICY (no CPU fallback)
---------------------------------
Per the project Hardware Target (NVIDIA Titan X Pascal) and an explicit
requirement, this engine runs **exclusively on the GPU and never falls back to
CPU**:
  * The embedding and reranking models are loaded on CUDA; if CUDA is not
    available, ``__init__`` raises immediately.
  * The FAISS index is moved to the GPU; if the installed FAISS build has no GPU
    support (e.g. ``faiss-cpu``), indexing raises immediately.
There is intentionally no ``try/except`` that degrades to CPU anywhere.

Note: ``faiss.normalize_L2`` operates in place on the host (numpy) array — it is
a data-preparation step on the embeddings, not model inference, so it does not
violate the GPU-only inference policy.

Public API
----------
``FinancialRetrievalEngine``
    ``__init__(embedding_model_name=..., reranker_model_name=...)``
    ``index_chunks(chunks)``
    ``query(query_str, top_k_dense=15, top_k_rerank=5)``
    ``save_index(path)`` / ``load_index(path)``

Runnable demo
-------------
``python -m src.retrieval_engine`` runs a smoke test on fake financial chunks.
It REQUIRES a CUDA GPU + a GPU-enabled FAISS build (it will not run on CPU).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import numpy as np
import faiss
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger("finalignrag.retrieval_engine")

# Metadata that must be carried through, verbatim, from input chunk to result.
REQUIRED_CHUNK_KEYS: tuple[str, ...] = (
    "chunk_id",
    "text",
    "ticker",
    "source_doc_id",
    "chunk_index",
)

# Filenames used by save_index / load_index.
_INDEX_FILE = "index.faiss"
_META_FILE = "meta.json"


def _require_cuda() -> None:
    """Fail hard unless a CUDA GPU is available (GPU-only, no CPU fallback)."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "FinancialRetrievalEngine is GPU-only (no CPU fallback), but "
            "torch.cuda.is_available() returned False. Run this on the CUDA GPU "
            "(e.g. the Titan X Pascal target). Refusing to fall back to CPU."
        )


def _require_gpu_faiss() -> None:
    """Fail hard unless the installed FAISS build exposes GPU support."""
    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError(
            "FinancialRetrievalEngine is GPU-only (no CPU fallback), but the "
            "installed FAISS build has no GPU support (looks like 'faiss-cpu'). "
            "Install a GPU FAISS build (e.g. `conda install -c pytorch faiss-gpu`). "
            "Refusing to build a CPU FAISS index."
        )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return a float32, C-contiguous, L2-normalized copy/view of ``matrix``.

    Uses ``faiss.normalize_L2`` (in place) exactly as required by the spec, so
    inner-product search over the index is equivalent to cosine similarity.
    """
    out = np.ascontiguousarray(matrix, dtype="float32")
    faiss.normalize_L2(out)
    return out


class FinancialRetrievalEngine:
    """Two-stage (dense + cross-encoder) retrieval engine for financial chunks."""

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        """Initialize the embedding and reranking models on the CUDA GPU.

        Raises
        ------
        RuntimeError
            If no CUDA GPU is available (GPU-only; no CPU fallback).
        """
        _require_cuda()
        self.device = "cuda"
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name

        logger.info("Loading embedding model '%s' on %s", embedding_model_name, self.device)
        self.embedder = SentenceTransformer(embedding_model_name, device=self.device)

        logger.info("Loading reranker model '%s' on %s", reranker_model_name, self.device)
        self.reranker = CrossEncoder(reranker_model_name, device=self.device)

        # Populated by index_chunks() / load_index().
        self.index: faiss.Index | None = None
        self._chunks: list[dict[str, Any]] = []
        self._dim: int | None = None
        # StandardGpuResources must outlive the GPU index; keep a reference.
        self._gpu_resources: Any = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _embed(self, texts: list[str]) -> np.ndarray:
        """Encode texts to a (n, dim) float32 array on the GPU."""
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,  # normalization is done via faiss.normalize_L2
        )
        return np.asarray(embeddings, dtype="float32")

    def _to_gpu_index(self, cpu_index: faiss.Index) -> faiss.Index:
        """Move a CPU FAISS index onto GPU 0 (GPU-only; raises if unavailable)."""
        _require_gpu_faiss()
        self._gpu_resources = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Encode chunk text into dense vectors and build a GPU FAISS index.

        Parameters
        ----------
        chunks:
            List of chunk dicts. Each must contain the required metadata keys
            (``chunk_id``, ``text``, ``ticker``, ``source_doc_id``,
            ``chunk_index``); all original fields are preserved for retrieval.

        Raises
        ------
        ValueError
            If ``chunks`` is empty or any chunk is missing a required key.
        RuntimeError
            If GPU FAISS support is unavailable.
        """
        if not chunks:
            raise ValueError("index_chunks() received an empty chunk list.")
        for i, chunk in enumerate(chunks):
            missing = [k for k in REQUIRED_CHUNK_KEYS if k not in chunk]
            if missing:
                raise ValueError(f"Chunk at position {i} missing required keys: {missing}")

        # Defensive copies so the engine never mutates the caller's chunks.
        self._chunks = [dict(c) for c in chunks]

        texts = [c["text"] for c in self._chunks]
        embeddings = self._embed(texts)
        embeddings = _l2_normalize(embeddings)  # doc-side L2 normalization
        self._dim = int(embeddings.shape[1])

        cpu_index = faiss.IndexFlatIP(self._dim)
        cpu_index.add(embeddings)
        self.index = self._to_gpu_index(cpu_index)

        logger.info(
            "Indexed %d chunks (dim=%d) into GPU IndexFlatIP", len(self._chunks), self._dim
        )

    def query(
        self,
        query_str: str,
        top_k_dense: int = 15,
        top_k_rerank: int = 5,
    ) -> list[dict[str, Any]]:
        """Dense-retrieve then cross-encoder-rerank; return <= ``top_k_rerank`` chunks.

        Each returned dict preserves all original chunk metadata and adds
        ``dense_score`` (cosine similarity from FAISS) and ``rerank_score``
        (cross-encoder logit). Results are sorted by ``rerank_score`` descending.

        Does not crash if fewer than ``top_k_dense``/``top_k_rerank`` candidates
        exist; it simply returns however many are available.

        Raises
        ------
        RuntimeError
            If called before ``index_chunks()`` / ``load_index()``.
        """
        if self.index is None:
            raise RuntimeError("query() called before an index was built/loaded.")

        # --- Stage 1: dense retrieval (query-side L2 normalization) ---
        query_vec = _l2_normalize(self._embed([query_str]))
        k_dense = min(top_k_dense, len(self._chunks))
        scores, indices = self.index.search(query_vec, k_dense)

        candidates: list[dict[str, Any]] = []
        for dense_score, row in zip(scores[0], indices[0]):
            if row < 0:  # FAISS pads with -1 when fewer results are available
                continue
            candidate = dict(self._chunks[row])  # preserve ALL metadata fields
            candidate["dense_score"] = float(dense_score)
            candidates.append(candidate)

        if not candidates:
            return []

        # --- Stage 2: cross-encoder reranking ---
        pairs = [(query_str, c["text"]) for c in candidates]
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
        for candidate, rerank_score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = float(rerank_score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k_rerank]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save_index(self, path: str) -> None:
        """Persist the FAISS index + chunk metadata to directory ``path``.

        The GPU index is copied back to CPU for serialization (a disk I/O step,
        not inference), then written alongside a JSON metadata sidecar.
        """
        if self.index is None or self._dim is None:
            raise RuntimeError("save_index() called before an index was built/loaded.")
        os.makedirs(path, exist_ok=True)

        cpu_index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(cpu_index, os.path.join(path, _INDEX_FILE))

        meta = {
            "dim": self._dim,
            "embedding_model_name": self.embedding_model_name,
            "reranker_model_name": self.reranker_model_name,
            "chunks": self._chunks,
        }
        with open(os.path.join(path, _META_FILE), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False)
        logger.info("Saved index (%d chunks) to %s", len(self._chunks), path)

    def load_index(self, path: str) -> None:
        """Load a FAISS index + chunk metadata from directory ``path`` onto GPU."""
        index_path = os.path.join(path, _INDEX_FILE)
        meta_path = os.path.join(path, _META_FILE)
        if not os.path.isfile(index_path) or not os.path.isfile(meta_path):
            raise FileNotFoundError(f"No saved index found under: {path}")

        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        self._dim = int(meta["dim"])
        self._chunks = meta["chunks"]

        cpu_index = faiss.read_index(index_path)
        self.index = self._to_gpu_index(cpu_index)
        logger.info("Loaded index (%d chunks) from %s onto GPU", len(self._chunks), path)


# ---------------------------------------------------------------------------
# Runnable demo / smoke test (GPU REQUIRED — will not run on CPU)
# ---------------------------------------------------------------------------
def _make_fake_chunks() -> list[dict[str, Any]]:
    """A handful of fake financial chunks across two tickers."""
    samples = [
        ("AAPL", "AAPL_2023_10K", 0,
         "Apple net income was 96.9 billion in 2023 versus 99.8 billion in 2022."),
        ("AAPL", "AAPL_2023_10K", 1,
         "Total net sales were 383.3 billion, a decline driven by lower iPhone revenue."),
        ("AAPL", "AAPL_2023_10K", 2,
         "Research and development expense increased to 29.9 billion in fiscal 2023."),
        ("MSFT", "MSFT_2023_10K", 0,
         "Microsoft revenue grew to 211.9 billion, up from 198.3 billion a year earlier."),
        ("MSFT", "MSFT_2023_10K", 1,
         "Operating income was 88.5 billion, reflecting strong cloud performance."),
        ("MSFT", "MSFT_2023_10K", 2,
         "Diluted earnings per share were 9.68 for the fiscal year."),
    ]
    return [
        {
            "chunk_id": f"{doc}_{idx:03d}",
            "text": text,
            "ticker": ticker,
            "source_doc_id": doc,
            "chunk_index": idx,
        }
        for ticker, doc, idx, text in samples
    ]


def _run_smoke_test(save_dir: str | None = None) -> None:
    """Index fake chunks, run a query, and assert metadata + normalization."""
    chunks = _make_fake_chunks()
    by_id = {c["chunk_id"]: c for c in chunks}

    engine = FinancialRetrievalEngine()
    engine.index_chunks(chunks)

    # --- Verify document-vector normalization actually happened ---
    cpu_index = faiss.index_gpu_to_cpu(engine.index)
    vec0 = cpu_index.reconstruct(0)
    norm0 = float(np.linalg.norm(vec0))
    assert abs(norm0 - 1.0) < 1e-3, f"Doc vector not L2-normalized (norm={norm0:.4f})"
    print(f"[OK] Document embeddings are L2-normalized (||v0|| = {norm0:.6f}).")

    # --- Run a query ---
    results = engine.query(
        "What was Microsoft's revenue compared to the prior year?",
        top_k_dense=15,
        top_k_rerank=3,
    )
    assert 0 < len(results) <= 3, f"Expected 1..3 results, got {len(results)}"
    print(f"[OK] query() returned {len(results)} result(s) (<= top_k_rerank).")

    # --- Verify scores present and cosine range (proves query normalization) ---
    for r in results:
        assert "dense_score" in r and "rerank_score" in r, "Missing score fields"
        assert -1.0001 <= r["dense_score"] <= 1.0001, (
            f"dense_score out of cosine range: {r['dense_score']}"
        )
    print("[OK] Every result includes dense_score (cosine range) and rerank_score.")

    # --- Verify metadata is perfectly preserved ---
    for r in results:
        original = by_id[r["chunk_id"]]
        for key in REQUIRED_CHUNK_KEYS:
            assert r[key] == original[key], (
                f"Metadata mismatch on '{key}': {r[key]!r} != {original[key]!r}"
            )
    print("[OK] Metadata (chunk_id, text, ticker, source_doc_id, chunk_index) preserved.")

    top = results[0]
    print(
        f"[TOP] {top['chunk_id']} ({top['ticker']}) "
        f"dense={top['dense_score']:.4f} rerank={top['rerank_score']:.4f}\n"
        f"      {top['text']}"
    )

    # --- Optional: exercise save/load round-trip ---
    if save_dir:
        engine.save_index(save_dir)
        reloaded = FinancialRetrievalEngine()
        reloaded.load_index(save_dir)
        results2 = reloaded.query(
            "What was Microsoft's revenue compared to the prior year?",
            top_k_dense=15,
            top_k_rerank=3,
        )
        assert results2[0]["chunk_id"] == top["chunk_id"], "save/load changed top result"
        print(f"[OK] save_index()/load_index() round-trip preserved top result.")

    print("\nSMOKE TEST PASSED.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.retrieval_engine",
        description="Smoke-test the GPU-only FinancialRetrievalEngine (requires CUDA + GPU FAISS).",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory to exercise save_index()/load_index() round-trip.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _run_smoke_test(save_dir=args.save_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
