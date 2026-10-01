# Document RAG Assistant

Question answering over PDF documents using hybrid retrieval (BM25 + vector search), cross-encoder reranking and LLM-based query expansion, with an evaluation workflow built on RAGAS. Answers come with the source chunks they were based on.

Two interfaces share one pipeline implementation (`rag_core.py`):

- **Streamlit app** (`app.py`): interactive demo UI
- **FastAPI service** (`api.py`): HTTP API, containerized with Docker

**Stack:** Python, LangChain, ChromaDB, BM25, Mistral AI (`mistral-small-2603` and Mistral embeddings), RAGAS, FastAPI, Streamlit, Docker

---

## Architecture

```
User question
    |
    v
Multi-query expansion (LLM rewrites the question into several variants)
    |
    v
Hybrid retrieval, for each variant:
    - BM25 (keyword search)    --+
    - Vector search (MMR)      --+--> merged with EnsembleRetriever (weights 0.4 BM25 / 0.6 vector)
    |
    v
Cross-encoder reranking (candidates re-scored, best 5 kept)
    |
    v
Strict, conflation-aware prompt --> Mistral LLM
    |
    v
Answer + cited source chunks
```

**Ingestion:** PDF, then `RecursiveCharacterTextSplitter` (1000 characters, 200 overlap), then Mistral embeddings, then a Chroma vector store. The chunks are also pickled separately (`bm25_store/chunks.pkl`), because BM25 needs the raw documents in memory and is not a persistent index like Chroma.

---

## Why each component is there

| Component | Problem it solves | Evidence |
| --- | --- | --- |
| **Hybrid retrieval** (BM25 + vector) | Pure vector search can miss exact terms, names and numbers that embeddings underweight | A query returned 7 merged candidates versus a fixed 4 from vector-only search |
| **Cross-encoder reranking** | Hybrid retrieval surfaced near-duplicate chunks (the same fact three times), wasting context slots | After reranking, duplicates were replaced with new information (a comparison table that had not made the original top 4) |
| **Multi-query expansion** | Vague or differently phrased questions can miss chunks that do not closely match the exact wording | A deliberately vague question ("How well did LSTM perform vs others?") went from a bare fact answer to a full synthesized comparison after expansion |
| **Strict, conflation-aware prompt** | Manual testing found two real generation failures (see below) | Both failures were fixed and verified against the same failing queries |

---

## Failures found during testing, and how they were fixed

Manual adversarial testing, not just happy-path queries, surfaced two hallucination-adjacent bugs:

**1. Metric conflation.** Asked "What was the LSTM model's precision score?", a metric never reported in the source document (which reports accuracy, R2, MAE, MSE and RMSE), the system answered "94%", reusing the accuracy figure under the wrong label. Fixed by adding an explicit instruction: answer only if the exact metric named is present in the context, and never substitute a similar-sounding one. Verified against the identical failing query.

**2. Cross-model number blending.** After retrieval was widened to fix a recall gap (see below), the model blended figures from different models in a comparison answer, for example attributing the Transformer's reported value to GRU. Fixed with an explicit per-model attribution instruction in the prompt. Verified with before/after evaluation runs.

---

## Evaluation

Automated evaluation with [RAGAS](https://github.com/explodinggradients/ragas) (`eval.py`), on a small 5-question set, scoring:

- **Faithfulness:** are the answer's claims supported by the retrieved context? (catches hallucination)
- **Answer relevancy:** does the answer address the question asked?
- **Context precision:** how much of the retrieved context was relevant?
- **Context recall:** did retrieval find what was needed to answer, compared with a reference answer?

### Iterative tuning (comparison question)

| Stage | Faithfulness | Context recall |
| --- | --- | --- |
| Baseline (k=8, top_n=4) | 1.00 | **0.50** (retrieval gap) |
| Widened retrieval (k=10, top_n=5) | **0.57** (new problem) | 1.00 |
| Widened retrieval + prompt fix (per-model attribution) | **1.00** | **1.00** |

Widening retrieval fixed the recall gap but introduced a faithfulness regression: more context gave the model more material to blend incorrectly. Retrieving more is not free. A targeted prompt fix resolved both problems, confirmed by re-running the evaluation. The baseline run is saved in `eval_results_before.csv` and the final run in `eval_results.csv`.

### Final results (5-question evaluation set)

| Question type | Faithfulness | Answer relevancy | Context precision | Context recall |
| --- | --- | --- | --- | --- |
| Fact lookup | 1.00 | 1.00 | 0.95 | 1.00 |
| Comparison synthesis | 1.00 | 0.85 | 1.00 | 1.00 |
| Open-ended summary | 1.00 | 0.99 | 1.00 | 1.00 |
| Correct refusal (metric not in document) | 0.00* | 0.00* | 0.92 | 1.00 |
| Correct refusal (out of scope) | 1.00 | 0.00* | 1.00 | 1.00 |

*Known metric limitation, not a system flaw: RAGAS faithfulness and relevancy work by decomposing an answer into factual claims. A correct refusal ("I could not find the answer") has no claims to check, and both metrics score it inconsistently across otherwise identical refusals (reproducible across runs). Manual verification confirmed both refusals were correct.

Run `python eval.py` to reproduce.

---

## Setup

1. Clone the repository and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your Mistral API key:
   ```
   MISTRAL_API_KEY=your_key_here
   ```
   The key is used for both the LLM (`mistral-small-2603`) and the embeddings.

Note: Mistral's free tier is rate-limited. If you hit limits during evaluation (which makes many LLM calls), use a paid key.

---

## Usage

There is no pre-build step. Documents are ingested at runtime.

**Streamlit app**

```
streamlit run app.py
```

Upload one or more PDFs in the UI, then ask questions.

**FastAPI service**

```
uvicorn api:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Health check |
| `POST /documents/process` | Upload PDFs (multipart field `files`) and build the indexes |
| `POST /ask` | Ask a question about the processed documents (request schema in `/docs`) |

The indexes are stored in `chroma_db/` (vectors) and `bm25_store/chunks.pkl` (BM25 chunks). Both are created automatically and are gitignored.

**Evaluation**

```
python eval.py
```

---

## Running with Docker

```
docker build -t rag-api .

# macOS / Linux
docker run -p 8000:8000 --env-file .env \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/bm25_store:/app/bm25_store \
  rag-api

# Windows PowerShell
docker run -p 8000:8000 --env-file .env `
  -v ${PWD}/chroma_db:/app/chroma_db `
  -v ${PWD}/bm25_store:/app/bm25_store `
  rag-api
```

The volumes keep the indexes between container restarts. The API is then available at http://localhost:8000/docs.

---

## Project structure

```
.
├── app.py                     # Streamlit demo UI
├── api.py                     # FastAPI service (Docker entry point)
├── rag_core.py                # Shared pipeline: ingestion, prompt, retrieval, generation
├── eval.py                    # RAGAS evaluation script
├── document_loaders/          # PDF and web loaders
├── retrievers/
│   ├── hybrid_retriever.py    # BM25 + vector (EnsembleRetriever)
│   ├── reranker.py            # Cross-encoder reranking
│   └── multi_query.py         # LLM-based query expansion
├── eval_results_before.csv    # Baseline evaluation run
├── eval_results.csv           # Final evaluation run
├── requirements.txt
├── Dockerfile
├── .env.example               # Environment variable template
└── README.md
```

---

## Notable debugging

- **`ragas` and `langchain-community` version conflict:** `ragas` imports a `ChatVertexAI` class from a `langchain_community` submodule that was removed in `langchain-community>=0.4.2`. Found by inspecting package internals across versions, and fixed by pinning `langchain-community==0.4.1`, the last version with the submodule, which is still compatible with the project's LangChain 1.x stack.
- **`langchain-mistralai` batching bug:** RAGAS's `answer_relevancy` metric generates several question paraphrases in one batched LLM call, and the token-usage merge logic in `langchain-mistralai` crashes when combining more than one result. Worked around by setting the metric's `strictness` to 1 (a single generation, so no merge is needed). This is a documented tradeoff, not a silent change.

---

## Limitations

- The evaluation set has 5 questions, so the scores show the effect of each change on those questions, not general benchmark performance.
- The tuning numbers above come from the comparison question specifically.
- The RAGAS faithfulness and relevancy metrics score correct refusals inconsistently (see the evaluation section).
- Index storage is local files (`chroma_db/`, `bm25_store/`), which suits a demo or a single-user service rather than a multi-tenant deployment.