"""
Multi-query retrieval: expands a single user question into several
reworded variants using the LLM, retrieves for each, and merges/dedupes
the results.

Why this matters:
- A single query embedding can miss relevant chunks if the user's exact
  phrasing doesn't closely match the document's phrasing. E.g. "What
  accuracy did LSTM get?" vs "How well did the LSTM model perform?" can
  retrieve slightly different chunks even though they're asking the same
  thing.
- Generating a few reworded variants and retrieving for each one widens
  the net BEFORE we narrow it back down with reranking. This mainly
  improves RECALL (are we finding all the relevant chunks), while
  reranking (already in place) improves PRECISION (are we keeping only
  the best of what we found).

Pipeline position: this wraps the hybrid retriever, so the full flow is:
    user query
      -> multi-query expansion (this file)
      -> hybrid retrieval (BM25 + vector) for EACH variant
      -> merged, deduped candidate set
      -> reranker.py narrows to the best top_n
      -> LLM

Cost note: this makes ~3-4x more retrieval calls per user question (one
per generated variant), plus one extra LLM call to generate the variants.
It's a reasonable tradeoff for answer quality, but worth knowing for
latency/cost discussions.
"""

try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    # Fallback for the langchain_classic split some newer installs use
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever


def get_multi_query_retriever(base_retriever, llm):
    """
    Wraps base_retriever (e.g. your hybrid BM25+vector retriever) so that
    the incoming query is first expanded into multiple variants by the
    LLM, each variant is retrieved against, and results are merged.
    """
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )