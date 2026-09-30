"""
FastAPI service for the RAG pipeline.

Why this exists (vs. just the Streamlit app):
- Streamlit is great for a demo/UI, but it's not something another
  service could call programmatically, and it doesn't give you HTTP
  endpoints, request/response schemas, or async handling.
- A FastAPI wrapper is the standard way to expose a RAG pipeline as a
  real backend service — e.g. if a separate frontend, a Slack bot, or
  another internal tool needed to query your documents.

Run with:
    uvicorn api:app --reload

Then visit http://127.0.0.1:8000/docs for interactive API docs
(FastAPI generates this automatically from the schemas below).
"""

import os
import shutil
from functools import lru_cache
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

import rag_core

load_dotenv()

app = FastAPI(
    title="Document RAG API",
    description="Upload PDFs and ask questions about them via hybrid "
                 "retrieval (BM25 + vector) with multi-query expansion "
                 "and cross-encoder reranking.",
    version="1.0.0",
)


# --------------------------------------------------
# Models are expensive to load, so cache them process-wide (equivalent
# to Streamlit's @st.cache_resource, but for a plain Python process).
# --------------------------------------------------

@lru_cache
def get_embedding_model():
    return rag_core.get_embedding_model()


@lru_cache
def get_llm():
    return rag_core.get_llm()


# --------------------------------------------------
# Request / response schemas
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    content: str
    page: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class ProcessResponse(BaseModel):
    documents_processed: int
    chunks_created: int


# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/health")
def health():
    """Basic liveness check."""
    return {"status": "ok"}


@app.post("/documents/process", response_model=ProcessResponse)
async def process_documents(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF files. Rebuilds the vector index and BM25
    store from scratch (same behavior as the Streamlit app's
    "Process Documents" button).
    """

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    rag_core.clear_directory(rag_core.UPLOAD_DIR)

    file_paths = []

    for uploaded_file in files:
        if not uploaded_file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are supported. Got: {uploaded_file.filename}",
            )

        file_path = os.path.join(rag_core.UPLOAD_DIR, uploaded_file.filename)
        contents = await uploaded_file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        file_paths.append(file_path)

    embedding_model = get_embedding_model()

    try:
        _, chunk_count = rag_core.process_documents(file_paths, embedding_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {e}")

    return ProcessResponse(
        documents_processed=len(file_paths),
        chunks_created=chunk_count,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Ask a question against the currently indexed documents.
    Requires /documents/process to have been called at least once.
    """

    if not os.path.exists(rag_core.CHROMA_DIR):
        raise HTTPException(
            status_code=400,
            detail="No documents have been processed yet. "
                   "Call /documents/process first.",
        )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    embedding_model = get_embedding_model()
    llm = get_llm()

    vectorstore = rag_core.get_vectorstore(embedding_model)
    retriever = rag_core.build_retriever(vectorstore, llm, k=10, top_n=5)

    try:
        answer, docs = rag_core.answer_question(request.question, retriever, llm)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

    sources = [
        SourceChunk(
            content=doc.page_content,
            page=(doc.metadata.get("page") + 1) if doc.metadata.get("page") is not None else None,
        )
        for doc in docs
    ]

    return AskResponse(answer=answer, sources=sources)