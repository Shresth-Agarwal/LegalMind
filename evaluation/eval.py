import json
import math
import numpy as np
from src.retriever import load_indexes, retrieve, dense_retrieve, sparse_retrieve
from src.reranker import load_reranker, rerank

chunks, bm25, faiss_index, embed_model = load_indexes()
tokenizer, rerank_model = load_reranker()

with open("eval_dataset.json") as f:
    dataset = json.load(f)

print(f"\nEvaluating on {len(dataset)} questions...\n")

# Metric functions

def recall_at_k(retrieved_indices: list[int], relevant: int, k: int) -> float:
    """
    1.0 if the relevant chunk appears in top-k results, else 0.0.

    Binary — either you found it or you didn't.
    This is the most important metric for RAG because if the
    right chunk isn't retrieved, no LLM can generate a good answer.
    """
    return 1.0 if relevant in retrieved_indices[:k] else 0.0


def reciprocal_rank(retrieved_indices: list[int], relevant: int) -> float:
    """
    1/rank if relevant chunk is found, else 0.

    Rewards systems that rank the right answer higher.
    MRR = mean of this across all questions.
    A system returning the right answer at rank 1 scores 1.0,
    at rank 2 scores 0.5, at rank 5 scores 0.2.
    """
    if relevant in retrieved_indices:
        rank = retrieved_indices.index(relevant) + 1
        return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_indices: list[int], relevant: int, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    Like MRR but uses a logarithmic discount — rank 1 is much
    better than rank 2, but rank 9 vs rank 10 barely matters.
    With a single relevant document, NDCG@k simplifies to:
      DCG  = 1 / log2(rank + 1)  if found in top-k, else 0
      IDCG = 1 / log2(1 + 1) = 1  (perfect score: rank 1)
      NDCG = DCG / IDCG
    """
    if relevant in retrieved_indices[:k]:
        rank = retrieved_indices[:k].index(relevant) + 1
        return 1.0 / math.log2(rank + 1)
    return 0.0


# Run one retrieval mode and collect metrics

def evaluate_mode(mode: str) -> dict:
    """
    Runs all questions through the specified retrieval mode
    and returns average metrics across the dataset.

    mode options:
      'dense'          — FAISS only
      'sparse'         — BM25 only
      'hybrid'         — RRF merged, no reranker
      'hybrid+rerank'  — RRF merged + cross-encoder reranker
    """
    r5_scores, r10_scores, mrr_scores, ndcg5_scores = [], [], [], []

    for item in dataset:
        query    = item["question"]
        relevant = item["relevant_chunk_index"]

        # Get ranked chunk indexes per mode
        if mode == "dense":
            indices = dense_retrieve(query, embed_model, faiss_index, k=10)

        elif mode == "sparse":
            indices = sparse_retrieve(query, bm25, k=10)

        elif mode == "hybrid":
            results = retrieve(query, chunks, bm25, faiss_index,
                               embed_model, top_k=10)
            indices = [r["chunk_index"] for r in results]

        elif mode == "hybrid+rerank":
            # Retrieve top 20 with hybrid, rerank down to top 5
            candidates = retrieve(query, chunks, bm25, faiss_index,
                                  embed_model, top_k=20)
            reranked   = rerank(query, candidates, tokenizer,
                                rerank_model, top_n=10)
            indices    = [r["chunk_index"] for r in reranked]

        # Compute metrics
        r5_scores.append(recall_at_k(indices, relevant, k=5))
        r10_scores.append(recall_at_k(indices, relevant, k=10))
        mrr_scores.append(reciprocal_rank(indices, relevant))
        ndcg5_scores.append(ndcg_at_k(indices, relevant, k=5))

    return {
        "Recall@5":  round(np.mean(r5_scores), 4),
        "Recall@10": round(np.mean(r10_scores), 4),
        "MRR":       round(np.mean(mrr_scores), 4),
        "NDCG@5":    round(np.mean(ndcg5_scores), 4),
    }


# Run all four modes

modes = ["dense", "sparse", "hybrid", "hybrid+rerank"]
results = {}

for mode in modes:
    print(f"Evaluating: {mode}...")
    results[mode] = evaluate_mode(mode)
    m = results[mode]
    print(f"  Recall@5={m['Recall@5']}  Recall@10={m['Recall@10']}  "
          f"MRR={m['MRR']}  NDCG@5={m['NDCG@5']}\n")


# Print comparison table

print("\n" + "="*65)
print(f"{'Method':<20} {'Recall@5':>10} {'Recall@10':>10} {'MRR':>8} {'NDCG@5':>8}")
print("="*65)

for mode in modes:
    m = results[mode]
    print(f"{mode:<20} {m['Recall@5']:>10} {m['Recall@10']:>10} "
          f"{m['MRR']:>8} {m['NDCG@5']:>8}")

print("="*65)

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✓ Results saved to eval_results.json")