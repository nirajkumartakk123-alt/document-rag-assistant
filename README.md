# Document RAG Assistant

A production-style Retrieval-Augmented Generation (RAG) system for
question-answering over PDF documents — built past the tutorial stage,
with hybrid retrieval, reranking, query expansion, and a measured,
iterated evaluation process.

Two interfaces to the same pipeline:
- **Streamlit app** (`app.py`) — interactive demo UI
- **FastAPI service** (`api.py`) — HTTP API for programmatic use

Both share a single pipeline implementation (`rag_core.py`), so there is
one source of truth for retrieval and generation logic.

---

## Architecture

```
User question
    │
    ▼
Multi-query expansion (LLM rewrites the question into several variants)
    │
    ▼
Hybrid retrieval — for each variant:
    • BM25 (keyword search)      ─┐
    • Vector search (MMR)        ─┴─► merged via EnsembleRetriever
    │
    ▼
Cross-encoder reranking (top candidates re-scored, best 5 kept)
    │
    ▼
Strict, conflation-aware prompt → Mistral LLM
    │
    ▼
Answer + cited source chunks
```

**Ingestion:** PDF → `RecursiveCharacterTextSplitter` (1000 chars,
200 overlap) → Mistral embeddings → Chroma vector store. Chunks are
also pickled separately (`bm25_store/`) since BM25 needs the raw
documents in memory and isn't a persistent index like Chroma.

---

## Why each component is there

| Component | Problem it solves | Evidence |
|---|---|---|
| **Hybrid retrieval** (BM25 + vector) | Pure vector search misses exact terms/names/numbers that embeddings underweight | Query returned 7 merged candidates vs. a fixed 4 from vector-only search |
| **Cross-encoder reranking** | Hybrid retrieval surfaced near-duplicate chunks (same fact repeated 3x), wasting context slots | After reranking, duplicate chunks were replaced with genuinely new information (a comparison table that hadn't made the original top-4) |
| **Multi-query expansion** | Vague or differently-phrased questions can miss chunks that don't closely match the query's exact wording | A deliberately vague question ("How well did LSTM perform vs others?") went from a bare fact answer to a full synthesized comparison after expansion |
| **Strict, conflation-aware prompt** | Manual testing found two real generation failures (see below) | Both failures fixed and verified against the same failing queries after the fix |

---

## Real failures found during testing, and how they were fixed

Manual adversarial testing (not just happy-path queries) surfaced two
genuine hallucination-adjacent bugs:

**1. Metric conflation.** Asked *"What was the LSTM model's precision
score?"* — a metric never reported in the source document (which only
reports accuracy, R², MAE, MSE, RMSE) — the system answered "94%,"
reusing the accuracy figure under the wrong label. Fixed by adding an
explicit instruction: only answer if the *exact* metric named is
present in context, never substitute a similar-sounding one. Verified
fixed against the identical failing query.

**2. Cross-model number blending.** When retrieval was widened (see
below) to fix a recall gap, a new problem appeared: the model blended
figures belonging to different models in a comparison answer (e.g.
attributing the Transformer's reported value to GRU). Fixed with an
explicit per-model attribution instruction in the prompt. Verified via
before/after evaluation runs (see results table).

---

## Evaluation

Automated evaluation via [RAGAS](https://github.com/explodinggradients/ragas)
(`eval.py`), scoring the pipeline on:
- **Faithfulness** — are the answer's claims actually supported by
  retrieved context? (catches hallucination)
- **Answer relevancy** — does the answer address the question asked?
- **Context precision** — how much of the retrieved context was
  actually relevant?
- **Context recall** — did retrieval find what was needed to answer,
  compared to a reference answer?

### Iterative tuning, with real before/after numbers

| Stage | Faithfulness (comparison Q) | Context Recall (comparison Q) |
|---|---|---|
| Baseline (k=8, top_n=4) | 1.00 | **0.50** ← retrieval gap |
| Widened retrieval (k=10, top_n=5) | **0.57** ← new problem | 1.00 |
| + Prompt fix (per-model attribution) | **1.00** | **1.00** |

Widening retrieval fixed the recall gap but introduced a faithfulness
regression — more context gave the model more material to blend
incorrectly. This is a real, measured precision/recall-style tradeoff
in RAG: retrieving more isn't free. A targeted prompt fix resolved
both simultaneously, confirmed by re-running the eval suite (numbers
above are from the final, re-verified run).

### Final results (5-question eval set)

| Question type | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|
| Fact lookup | 1.00 | 1.00 | 0.95 | 1.00 |
| Comparison synthesis | 1.00 | 0.85 | 1.00 | 1.00 |
| Open-ended summary | 1.00 | 0.99 | 1.00 | 1.00 |
| Correct refusal (metric not in doc) | 0.00* | 0.00* | 0.92 | 1.00 |
| Correct refusal (out of scope) | 1.00 | 0.00* | 1.00 | 1.00 |

**\*Known metric limitation, not a system flaw:** RAGAS's faithfulness
and relevancy metrics work by decomposing the answer into factual
claims to check. A correct refusal ("I could not find the answer")
has no claims to check, and both metrics score this inconsistently
across otherwise-identical refusal answers (confirmed reproducible
across multiple runs). Manual verification confirms both refusals
were factually correct.

Run `python eval.py` to reproduce these results.

---

## Notable debugging along the way

- **`ragas` / `langchain-community` version conflict:** `ragas` imports
  a `ChatVertexAI` class from a `langchain_community` submodule that
  was removed in `langchain-community>=0.4.2`. Root-caused via direct
  inspection of package internals across versions; fixed by pinning to
  `langchain-community==0.4.1` (last version with the submodule, still
  compatible with the rest of the project's newer LangChain 1.x stack).
- **`langchain-mistralai` batching bug:** RAGAS's `answer_relevancy`
  metric generates multiple question paraphrases in one batched LLM
  call; `langchain-mistralai`'s token-usage merge logic crashes trying
  to add nested dicts together when combining >1 result. Worked around
  by reducing the metric's `strictness` to 1 (single generation, no
  merge needed) — a real, documented tradeoff, not a silent hack.

---

## Usage

There is no pre-build step. Documents are ingested at runtime.

**Streamlit app:** run `streamlit run app.py`, upload one or more PDFs
in the UI, then ask questions.

**API:** start the server (`uvicorn api:app --reload`), then:
1. `POST /documents/process` with your PDF files (multipart field `files`)
2. `POST /ask` with your question
Interactive docs: http://127.0.0.1:8000/docs

The index is stored in `chroma_db/` (vectors) and `bm25_store/chunks.pkl`
(BM25 chunks). Both are created automatically and are gitignored.
In Docker, the mounted volumes keep the index between restarts.

## Running locally

```bash
# Install dependencies
pip install -r requirements.txt

# Streamlit demo
streamlit run app.py

# FastAPI service
uvicorn api:app --reload
# then visit http://127.0.0.1:8000/docs

# Evaluation
python eval.py
```

## Running with Docker (API service)

```bash
docker build -t rag-api .
docker run -p 8000:8000 --env-file .env \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/bm25_store:/app/bm25_store \
  rag-api
```

---

## Project structure

```
.
├── app.py                     # Streamlit demo UI
├── api.py                     # FastAPI service
├── rag_core.py                # Shared pipeline: ingestion, prompt, retrieval, generation
├── eval.py                    # RAGAS evaluation script
├── retrievers/
│   ├── hybrid_retriever.py    # BM25 + vector (EnsembleRetriever)
│   ├── reranker.py            # Cross-encoder reranking
│   └── multi_query.py         # LLM-based query expansion
├── requirements.txt
├── Dockerfile
└── eval_results.csv           # Latest evaluation run output
```

## Setup
1. Copy `.env.example` to `.env`
2. Add your `MISTRAL_API_KEY` (used for the LLM and embeddings)
Note: Mistral's free tier is rate-limited. If you hit limits during evaluation, use a paid API key.