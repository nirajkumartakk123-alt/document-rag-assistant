"""
Shared RAG pipeline core.

Why this file exists:
- Up to now, app.py (Streamlit) and eval.py (evaluation script) each had
  their own copy of the prompt and retriever-building logic. Every time
  we fixed something (e.g. the metric-conflation prompt fix), it had to
  be manually copied into both files — a real source of drift and bugs.
- This module is the single source of truth for: the embedding model,
  the LLM, the system prompt, and how the full retrieval pipeline
  (multi-query -> hybrid -> reranking) is assembled. app.py, eval.py,
  and api.py all import from here instead of duplicating logic.
"""

import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from retrievers.hybrid_retriever import save_chunks_for_bm25, get_hybrid_retriever
from retrievers.reranker import get_reranked_retriever
from retrievers.multi_query import get_multi_query_retriever


CHROMA_DIR = "chroma_db"
UPLOAD_DIR = "uploaded_documents"
COLLECTION_NAME = "rag_documents"


def clear_directory(path):
    """
    Removes everything INSIDE a directory, but not the directory itself.

    Why this matters: when running in Docker, chroma_db (and potentially
    uploaded_documents) are mounted volumes. shutil.rmtree() on the
    directory itself fails with "Device or resource busy" on a mount
    point — you can clear a mounted directory's contents, but you can't
    delete the mount point. This function does the former safely,
    whether or not the directory happens to be a mount point.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return

    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path)
        else:
            os.remove(entry_path)


# --------------------------------------------------
# Models (no Streamlit caching here — callers that run under Streamlit
# should wrap these with @st.cache_resource themselves; api.py and
# eval.py just call them directly and hold the result).
# --------------------------------------------------

def get_embedding_model():
    return MistralAIEmbeddings()


def get_llm():
    return ChatMistralAI(model="mistral-small-2603")


# --------------------------------------------------
# Prompt — the single source of truth. Includes both fixes discovered
# during evaluation: (1) strict exact-metric matching, to prevent
# substituting a similar-sounding metric (e.g. "precision" answered
# with an "accuracy" value), and (2) precise per-model attribution, to
# prevent blending numbers across different models/entities when
# synthesizing a comparison.
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Answer the user's question using ONLY the
provided context.

Be strict about matching the EXACT thing the user
asked for. If the user asks about a specific metric,
term, or quantity (e.g. "precision", "recall",
"F1 score"), only answer if that EXACT metric or
term is explicitly named in the context. Do not
substitute a different metric just because it is
nearby, related, or has a similar-looking value
(for example, do not answer a question about
"precision" using a number that the context labels
as "accuracy" — these are different metrics, even
if the number looks similar).

If the exact thing asked about is not explicitly
present in the context, say:

"I could not find the answer in the document."

When comparing multiple models, methods, or entities,
be precise about which specific number belongs to
which specific one. Do not attribute a value stated
for one model (e.g. Transformer) to a different model
(e.g. GRU) just because they appear in the same
sentence or paragraph. Do not blend, average, or
cross-apply numbers between different models or
different metrics. If the context's wording makes it
ambiguous which model or metric a number refers to,
say so explicitly rather than guessing.

Do not use outside knowledge. Do not guess or infer
values that are not explicitly stated.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """
        ),
    ]
)


# --------------------------------------------------
# Ingestion
# --------------------------------------------------

def process_documents(file_paths, embedding_model):
    """
    file_paths: list of paths to PDF files already saved to disk
    (both Streamlit and FastAPI save uploads to disk first, then call
    this the same way).

    Rebuilds the Chroma index and the BM25 chunk store from scratch.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    all_chunks = []

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    if os.path.exists(CHROMA_DIR):
        clear_directory(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    save_chunks_for_bm25(all_chunks)

    return vectorstore, len(all_chunks)


def get_vectorstore(embedding_model):
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )


# --------------------------------------------------
# Full retrieval pipeline: multi-query -> hybrid -> reranking
# --------------------------------------------------

def build_retriever(vectorstore, llm, k=10, top_n=5):
    base_retriever = get_hybrid_retriever(vectorstore, k=k)
    expanded_retriever = get_multi_query_retriever(base_retriever, llm)
    retriever = get_reranked_retriever(expanded_retriever, top_n=top_n)
    return retriever


# --------------------------------------------------
# End-to-end question answering
# --------------------------------------------------

def answer_question(question, retriever, llm):
    """Returns (answer_text, list_of_source_documents)."""

    docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({"context": context, "question": question})

    response = llm.invoke(final_prompt)

    return response.content, docs