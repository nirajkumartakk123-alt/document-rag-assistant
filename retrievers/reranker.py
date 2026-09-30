"""
Cross-encoder reranking.

Why this matters:
- Vector search and BM25 both score query-vs-chunk similarity FAST but
  approximately — they embed the query and each chunk independently, then
  compare vectors. This misses a lot of nuance and often ranks
  near-duplicate or tangential chunks too highly (exactly what you saw:
  the same "94% accuracy" sentence showing up 3 times in your results).

- A cross-encoder reranker instead looks at the query and each candidate
  chunk TOGETHER (jointly), and outputs a relevance score. This is far
  more accurate at judging "does this chunk actually answer the question"
  — but it's slower, so it's only practical on a small shortlist (e.g. the
  top 6-8 chunks from hybrid search), not the whole document store.

Standard production pattern:
  1. Hybrid retriever pulls a wider candidate set (e.g. top 8)
  2. Cross-encoder reranker re-scores those 8 against the query
  3. Only the top N (e.g. top 4) get passed to the LLM as context

This is wrapped as a LangChain ContextualCompressionRetriever, so it
slots in transparently wherever you already call retriever.invoke(query).
"""

try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    # Fallback for the langchain_classic split some newer installs use
    from langchain_classic.retrievers import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from langchain_community.cross_encoders import HuggingFaceCrossEncoder


_cross_encoder_model = None


def _get_cross_encoder():
    """Load the cross-encoder model once and reuse it (it's not tiny to
    load, so we don't want to reload it on every query)."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        _cross_encoder_model = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    return _cross_encoder_model


def get_reranked_retriever(base_retriever, top_n=5):
    """
    Wraps any retriever (e.g. your hybrid BM25+vector retriever) with
    cross-encoder reranking. The base_retriever should return a somewhat
    wider candidate set (e.g. k=8) so the reranker has something
    meaningful to choose from; this function then narrows it to top_n.
    """
    cross_encoder = _get_cross_encoder()

    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

    return compression_retriever