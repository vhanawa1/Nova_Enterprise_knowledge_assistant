"""
Retrieval + reranking.

Step 1 - Dense retrieval: semantic vector search via ChromaDB (top TOP_K_RETRIEVE).
Step 2 - Hybrid fusion (optional): re-score candidates by blending dense
          cosine similarity with a sparse BM25 score computed over the same
          candidate set. This helps when the query contains exact terms
          (policy names, error codes, form numbers) that pure embeddings
          can under-weight.
Step 3 - Rerank: cross-encoder rerank (sentence-transformers) for precision
          if installed; otherwise falls back to the hybrid score order.
          Cross-encoders score (query, chunk) pairs jointly, which is far
          more accurate than embedding cosine similarity for final ranking.
Step 4 - Truncate to TOP_K_RERANK chunks that get passed to the LLM.
"""
from __future__ import annotations

from typing import Any

import config
from rag.vector_store import VectorStore

_cross_encoder = None  # lazy-loaded singleton


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"[WARN] Cross-encoder unavailable, falling back to hybrid score: {e}")
            _cross_encoder = False
    return _cross_encoder


def _bm25_scores(query: str, candidates: list[dict[str, Any]]) -> list[float]:
    from rank_bm25 import BM25Okapi
    corpus = [c["text"].lower().split() for c in candidates]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    # normalize to 0-1
    if max(scores, default=0) > 0:
        scores = [s / max(scores) for s in scores]
    return list(scores)


def retrieve_and_rerank(
    query: str,
    vector_store: VectorStore,
    where: dict | None = None,
    top_k_retrieve: int = config.TOP_K_RETRIEVE,
    top_k_final: int = config.TOP_K_RERANK,
) -> list[dict[str, Any]]:
    candidates = vector_store.query(query, top_k=top_k_retrieve, where=where)
    if not candidates:
        return []

    # --- Hybrid fusion ---
    if config.USE_HYBRID_SEARCH and len(candidates) > 1:
        try:
            sparse_scores = _bm25_scores(query, candidates)
            for c, s in zip(candidates, sparse_scores):
                c["hybrid_score"] = config.HYBRID_ALPHA * c["score"] + (1 - config.HYBRID_ALPHA) * s
        except Exception as e:
            print(f"[WARN] BM25 fusion failed, using dense score only: {e}")
            for c in candidates:
                c["hybrid_score"] = c["score"]
    else:
        for c in candidates:
            c["hybrid_score"] = c["score"]

    # --- Cross-encoder rerank ---
    encoder = _get_cross_encoder()
    if encoder:
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = encoder.predict(pairs)
        # squash to 0-1 via min-max for consistent confidence scoring
        lo, hi = min(rerank_scores), max(rerank_scores)
        span = (hi - lo) or 1.0
        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float((s - lo) / span)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        final_score_key = "rerank_score"
    else:
        candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)
        final_score_key = "hybrid_score"

    top = candidates[:top_k_final]
    for c in top:
        c["final_score"] = c[final_score_key]
    return top
