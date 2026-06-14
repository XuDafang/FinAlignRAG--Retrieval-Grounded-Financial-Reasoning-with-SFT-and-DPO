"""FinAlignRAG source package.

Retrieval-Grounded Financial Reasoning with SFT and DPO.

Modules
-------
data_pipeline      : ingestion, chunking, split-by-ticker leakage control (Step 1)
retrieval_engine   : FAISS dense retrieval + cross-encoder reranking (Step 2)
eval_harness       : deterministic JSON / numerical / evidence scoring (Step 3)
alignment          : QLoRA SFT + DPO training (Step 4)
rag_pipeline       : RAG prompt serialization & inference coordination
"""

__all__ = [
    "data_pipeline",
    "retrieval_engine",
    "eval_harness",
    "alignment",
    "rag_pipeline",
]
