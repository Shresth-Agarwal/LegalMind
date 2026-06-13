from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_N        = 5   # final results returned after reranking

# Load reranker

def load_reranker():
    """
    Loads a cross-encoder model fine-tuned on MS MARCO passage ranking.

    Why a cross-encoder and not another bi-encoder?
    Bi-encoders (like MiniLM used in FAISS) encode query and
    document separately — fast, but they never "see" each other
    during encoding. Cross-encoders take the (query, document)
    pair as a single input, so the model can compute full
    attention between every query token and every document token.
    This gives much better relevance scores at the cost of speed.

    Why MS MARCO?
    It's a large passage retrieval dataset from Bing search logs.
    Cross-encoders trained on it generalise well to legal text
    because the task is the same: given a question, which passage
    answers it best?
    """
    print(f"Loading reranker: {RERANK_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
    model.eval()
    print("  → Reranker ready")
    return tokenizer, model

# Rerank

def rerank( query: str, chunks: list[dict], tokenizer, model, top_n: int = TOP_N) -> list[dict]:
    """
    Scores each (query, chunk_text) pair and returns top_n chunks
    sorted by reranker score descending.

    The cross-encoder outputs a single logit per pair.
    Higher = more relevant. No softmax needed — we only care
    about relative ordering, not calibrated probabilities.
    """
    if not chunks:
        return []

    pairs = [(query, c["text"]) for c in chunks]

    # Tokenize all pairs in one batch
    features = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        scores = model(**features).logits.squeeze(-1)  # shape: (num_chunks,)

    # Attach scores to chunks
    scored = []
    for i, chunk in enumerate(chunks):
        scored.append({
            **chunk,
            "rerank_score": scores[i].item()
        })

    # Sort by reranker score descending
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_n]

if __name__ == "__main__":
    from retriever import load_indexes, retrieve

    chunks, bm25, faiss_index, embed_model = load_indexes()
    tokenizer, rerank_model = load_reranker()

    test_queries = [
        "What is the rarest of rare doctrine for death penalty?",
        "Section 302 punishment for murder",
        "culpable homicide not amounting to murder",
        "arrest guidelines under Section 498A"
    ]

    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query}")
        print(f"{'─'*60}")

        # Step 1: hybrid retrieve top 20
        candidates = retrieve(query, chunks, bm25, faiss_index, embed_model, top_k=20)

        # Step 2: rerank down to top 5
        final = rerank(query, candidates, tokenizer, rerank_model)

        for i, r in enumerate(final, 1):
            print(f"\n[{i}] Score: {r['rerank_score']:.4f} | Source: {r['source']}")
            print(f"    {r['text'][:200]}...")