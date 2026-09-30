"""
Automated evaluation for the RAG pipeline, using RAGAS.

This script now pulls the prompt and retrieval pipeline from rag_core.py
instead of maintaining its own copy — so evaluation always tests the
exact same logic the live app uses, with no risk of the two drifting
out of sync (which happened earlier in this project when the prompt
was fixed in app.py but eval.py still tested the old version).

Run: python eval.py   (or: uv run python eval.py)
Requires: pip install ragas datasets
"""

from dotenv import load_dotenv

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import rag_core

load_dotenv()

# Workaround for a langchain-mistralai bug: when answer_relevancy generates
# multiple rephrased questions per answer (default strictness=3) in one
# batched call, langchain-mistralai's token-usage merge logic crashes
# trying to add nested dicts together (TypeError: dict += dict). Reducing
# strictness to 1 means only a single generation happens per call, so
# there's nothing to merge and the buggy code path never runs.
answer_relevancy.strictness = 1


# --------------------------------------------------
# Eval set: (question, ground_truth) pairs, built from manually-verified
# test cases run against analysis.pdf (the LSTM/GRU/Transformer paper).
# --------------------------------------------------

EVAL_SET = [
    {
        "question": "What accuracy did the LSTM model achieve?",
        "ground_truth": "The LSTM model achieved an accuracy of 94%.",
    },
    {
        "question": "How well did the LSTM model perform as compared to the others?",
        "ground_truth": (
            "The LSTM model performed the best among LSTM, GRU, and "
            "Transformer models, achieving 94% accuracy and outperforming "
            "the other two models based on R2, MAE, MSE, and RMSE metrics."
        ),
    },
    {
        "question": "What are the main findings of this paper?",
        "ground_truth": (
            "The LSTM model achieved the highest prediction accuracy (94%) "
            "among LSTM, GRU, and Transformer models tested for Tesla stock "
            "price forecasting, outperforming the other two on R2, MAE, MSE, "
            "and RMSE metrics."
        ),
    },
    {
        "question": "What was the LSTM model's precision score?",
        "ground_truth": (
            "The document does not report a precision score for the LSTM "
            "model. It reports accuracy (94%), R2, MAE, MSE, and RMSE "
            "instead."
        ),
    },
    {
        "question": "What programming language was used to build this model?",
        "ground_truth": (
            "The document does not specify which programming language was "
            "used."
        ),
    },
]


def main():

    print("Loading models and vector store...")

    embedding_model = rag_core.get_embedding_model()
    llm = rag_core.get_llm()

    vectorstore = rag_core.get_vectorstore(embedding_model)

    print("Building retrieval pipeline (multi-query -> hybrid -> reranker)...")

    retriever = rag_core.build_retriever(vectorstore, llm, k=10, top_n=5)

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, item in enumerate(EVAL_SET):
        question = item["question"]
        print(f"\n[{i + 1}/{len(EVAL_SET)}] Running: {question}")

        answer, docs = rag_core.answer_question(question, retriever, llm)
        contexts = [doc.page_content for doc in docs]

        print(f"  Answer: {answer[:120]}...")

        questions.append(question)
        answers.append(answer)
        contexts_list.append(contexts)
        ground_truths.append(item["ground_truth"])

    print("\nRunning RAGAS evaluation...")

    eval_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        }
    )

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=True,
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(result)

    df = result.to_pandas()
    df.to_csv("eval_results.csv", index=False)
    print("\nDetailed per-question results saved to eval_results.csv")


if __name__ == "__main__":
    main()