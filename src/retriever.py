import json
import pickle
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss


INDEX_DIR  = Path("indexes")
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 20   # how many results each retriever returns before merging

# Load indexes

def load_indexes():
    """
    Loads all three artifacts built during ingestion.
    Called once at startup — not on every query.
    """
    print("Loading indexes...")

    with open(INDEX_DIR / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    faiss_index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    model       = SentenceTransformer(EMBED_MODEL)

    print(f"  → {len(chunks)} chunks loaded")
    return chunks, bm25, faiss_index, model


# Dense retrieval

def dense_retrieve(query: str, model, faiss_index, k: int = TOP_K) -> list[int]:
    """
    Embeds the query and searches FAISS for the k nearest vectors.

    Returns a list of chunk indexes (positions in chunks.json),
    ordered by L2 distance — closest first.

    Why L2 and not cosine?
    MiniLM embeddings are not normalized by default, so L2 and
    cosine give slightly different rankings. For this project
    the difference is negligible. If you wanted cosine, you'd
    use faiss.IndexFlatIP after normalizing vectors with
    faiss.normalize_L2().
    """
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = faiss_index.search(query_vec, k)
    return indices[0].tolist()


# Sparse retrieval

def sparse_retrieve(query: str, bm25, k: int = TOP_K) -> list[int]:
    """
    Tokenizes the query and scores every chunk using BM25Okapi.

    Returns the indexes of the top-k scoring chunks.

    Why does sparse retrieval matter here?
    If someone queries "Section 302 punishment", the word '302'
    is a rare token. Dense embeddings might rank a chunk about
    Section 304 (culpable homicide) equally high because the
    surrounding legal language is similar. BM25 will strongly
    prefer chunks that literally contain '302'.
    """
    tokenized_query = query.lower().split()
    scores          = bm25.get_scores(tokenized_query)  # array of len(chunks)
    top_k_indices   = np.argsort(scores)[::-1][:k]      # descending
    return top_k_indices.tolist()


# Reciprocal Rank Fusion

def reciprocal_rank_fusion(dense_indices: list[int], sparse_indices: list[int], k: int = 60) -> list[int]:
    """
    Merges two ranked lists into one using Reciprocal Rank Fusion.

    RRF score for a chunk = sum of 1/(k + rank) across all lists
    where rank is 1-indexed position in that list.

    Why k=60?
    It's the standard RRF constant from the original 2009 paper.
    It dampens the influence of very high ranks (rank 1, 2, 3)
    so that a chunk ranked #1 in one list doesn't completely
    dominate a chunk ranked #1 in both lists.

    Why RRF and not a weighted average of scores?
    Scores from FAISS (L2 distances) and BM25 live on completely
    different scales — you can't average them meaningfully.
    RRF only uses rank positions, which are scale-free.
    This is the key insight behind hybrid search.
    """
    scores = {}

    for rank, idx in enumerate(dense_indices, start=1):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

    for rank, idx in enumerate(sparse_indices, start=1):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank)

    # Sort by RRF score descending
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in merged]


# Main retrieve function

def retrieve(query: str, chunks: list[dict], bm25, faiss_index, model, top_k: int = TOP_K) -> list[dict]:
    """
    Full hybrid retrieval pipeline.
    Returns top_k chunks with their source metadata.
    """
    dense_indices  = dense_retrieve(query, model, faiss_index, k=top_k)
    sparse_indices = sparse_retrieve(query, bm25, k=top_k)
    merged_indices = reciprocal_rank_fusion(dense_indices, sparse_indices)

    results = []
    for idx in merged_indices[:top_k]:
        chunk = chunks[idx].copy()
        chunk["chunk_index"] = idx
        results.append(chunk)

    return results

if __name__ == "__main__":
    chunks, bm25, faiss_index, model = load_indexes()

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

        results = retrieve(query, chunks, bm25, faiss_index, model)

        for i, r in enumerate(results[:3], 1):
            print(f"\n[{i}] Source: {r['source']}")
            print(f"    {r['text'][:200]}...")