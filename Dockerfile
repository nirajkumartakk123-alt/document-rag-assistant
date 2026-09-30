# Production image for the RAG FastAPI service.
# Note: this containerizes api.py (the FastAPI backend), not app.py
# (the Streamlit demo UI) — those are typically deployed separately.

FROM python:3.12-slim

WORKDIR /app

# System deps needed by some ML libraries (sentence-transformers, torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Persisted data (chroma_db, bm25_store, uploaded_documents) should be
# mounted as a volume in production so it survives container restarts:
#   docker run -v $(pwd)/data:/app/chroma_db ...
VOLUME ["/app/chroma_db", "/app/bm25_store", "/app/uploaded_documents"]

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
