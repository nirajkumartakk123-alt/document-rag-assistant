"""
Hybrid retriever: combines BM25 (keyword) search with vector (semantic) search.

Why this matters:
- Vector search finds semantically similar chunks but can miss exact terms
  (names, codes, acronyms, numbers) if the embedding doesn't weight them heavily.
- BM25 is a classic keyword-matching algorithm that's very good at exactly
  the things vector search is weak at.
- Running both and merging results (via EnsembleRetriever) generally beats
  either one alone. This is the standard "hybrid search" pattern used in
  most production RAG systems.

BM25Retriever needs the full list of chunk Documents in memory (it's not a
persistent index the way Chroma is), so we pickle the chunks alongside the
Chroma DB whenever we (re)process documents, and reload them here.
"""

import os
import pickle

from langchain_community.retrievers import BM25Retriever

try:
    # Standard location in most langchain versions
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    # Fallback for the langchain_classic split some newer installs use
    from langchain_classic.retrievers.ensemble import EnsembleRetriever


BM25_STORE_DIR = "bm25_store"
BM25_STORE_PATH = os.path.join(BM25_STORE_DIR, "chunks.pkl")


def save_chunks_for_bm25(chunks):
    """Call this once, right after you build/rebuild the Chroma vectorstore,
    passing the same `chunks` list you indexed. This persists the raw
    chunk text+metadata so BM25 can be rebuilt without re-parsing PDFs."""
    os.makedirs(BM25_STORE_DIR, exist_ok=True)
    with open(BM25_STORE_PATH, "wb") as f:
        pickle.dump(chunks, f)


def _load_chunks_for_bm25():
    if not os.path.exists(BM25_STORE_PATH):
        raise FileNotFoundError(
            "No BM25 chunk store found. Process documents at least once "
            "before asking questions."
        )
    with open(BM25_STORE_PATH, "rb") as f:
        return pickle.load(f)


def get_hybrid_retriever(vectorstore, k=10, vector_weight=0.6, bm25_weight=0.4):
    """
    Returns an EnsembleRetriever that merges:
      - BM25Retriever (keyword search)
      - vectorstore's MMR retriever (semantic search)

    vector_weight / bm25_weight control how much each contributes to the
    final ranking. 0.6/0.4 is a reasonable starting point; tune later
    once you have an eval set (Day 7).
    """
    chunks = _load_chunks_for_bm25()

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )

    return hybrid_retriever