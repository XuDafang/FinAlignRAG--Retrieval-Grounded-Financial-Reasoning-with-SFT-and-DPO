"""FinAlignRAG source package.

Retrieval-Grounded Financial Reasoning with SFT and DPO.

Modules
-------
build_sft_data     : FinQA extraction and held-out evaluation preparation
gen_dpo_data       : preference-pair generation from SFT targets
alignment          : QLoRA SFT and DPO training
retrieval_engine   : FAISS dense retrieval and cross-encoder reranking
rag_pipeline       : RAG prompt serialization & inference coordination
eval_harness       : deterministic JSON, numerical, and evidence scoring
"""

__all__ = [
    "build_sft_data",
    "gen_dpo_data",
    "alignment",
    "retrieval_engine",
    "rag_pipeline",
    "eval_harness",
]
