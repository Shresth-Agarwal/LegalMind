# LegalMind — Hybrid RAG Pipeline for Indian Legal Documents

A retrieval-augmented generation pipeline built from first principles over Indian Penal Code sections and Supreme Court judgments. Implements hybrid search (dense + sparse), Reciprocal Rank Fusion, cross-encoder reranking, and quantitative evaluation across four retrieval configurations.

---

## Why This Project Exists

Managed RAG services (like AWS Bedrock Knowledge Base) abstract away the retrieval layer. This project builds that layer from scratch — every component is explicit, measurable, and replaceable.

The domain is Indian legal text: IPC sections and 20 landmark Supreme Court judgments. Legal text is a hard retrieval problem — dense with jargon, full of exact statute references, and semantically rich. It breaks naive retrieval systems in interesting ways.

---

## Architecture

```
Query
  ├── Dense Retrieval    →  FAISS (all-MiniLM-L6-v2 embeddings)
  ├── Sparse Retrieval   →  BM25Okapi (exact term matching)
  └── RRF Merge          →  Reciprocal Rank Fusion (rank-based, scale-free)
        └── Reranker     →  cross-encoder/ms-marco-MiniLM-L-6-v2
              └── Top 5 results → FastAPI response
```

**Why hybrid search?**
Dense retrieval finds semantically similar chunks — "person killed intentionally" matches murder sections even without those exact words. Sparse retrieval finds exact references — "Section 302" or "Arnesh Kumar" must match literally. Neither alone is sufficient for legal text. RRF combines both ranked lists without needing to reconcile their incompatible score scales.

---

## Corpus

| Type | Source | Count |
|---|---|---|
| IPC Bare Act | indiacode.nic.in | 1 PDF, ~312 chunks |
| SC Judgments | indiankanoon.org | 20 PDFs, ~2305 chunks |
| **Total** | | **2617 chunks** |

Judgments span: murder & homicide (IPC 302), sexual offences (IPC 376), fraud & cheating (IPC 420), criminal conspiracy (IPC 120B), domestic cruelty (IPC 498A), theft & dacoity (IPC 379/395), and constitutional landmarks.

---

## Evaluation Results

Evaluated on 25 hand-labelled query–chunk pairs across all IPC topics.

| Method | Recall@5 | Recall@10 | MRR | NDCG@5 |
|---|---|---|---|---|
| Dense only | 0.667 | 0.867 | 0.482 | 0.510 |
| Sparse only | 0.467 | 0.800 | 0.387 | 0.371 |
| Hybrid (RRF) | **1.000** | **1.000** | **0.593** | **0.694** |
| Hybrid + Rerank | 0.800 | 0.867 | 0.519 | 0.584 |

**Key findings:**

**Hybrid RRF achieves perfect Recall@5** — every correct answer appeared in the top 5 results. Dense alone missed 33% of answers at rank 5; sparse alone missed 53%. RRF's rank-based fusion covered each retriever's blind spots completely.

**The reranker reduced recall** — the cross-encoder (`ms-marco-MiniLM-L-6-v2`) was trained on MS MARCO web search data. Indian legal text has a fundamentally different language pattern, causing domain mismatch. In production this would be addressed by fine-tuning the cross-encoder on legal query-passage pairs.

---

## Stack

| Component | Library | Purpose |
|---|---|---|
| Embeddings | `sentence-transformers` | Dense vector encoding |
| Vector search | `faiss-cpu` | Approximate nearest neighbour |
| Sparse search | `rank_bm25` | BM25Okapi term matching |
| Reranker | `transformers` + PyTorch | Cross-encoder scoring |
| API | `FastAPI` | Query endpoint |
| PDF parsing | `pypdf` | Text extraction |

---

## Project Structure

```
LegalMind/
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   ├── routers.py              # API routes
│   └── schemas.py              # Request/response schemas
├── data/
│   ├── ipc_act.pdf
│   └── judgements/         # 20 SC judgment PDFs
├── evaluation/
│   ├── build_eval.py           # Build evaluation dataset
│   ├── eval.py                 # Run retrieval evaluation
│   ├── eval_dataset.json       # Evaluation queries & ground truth
│   └── eval_results.json       # Evaluation metrics output
├── indexes/
│   ├── faiss.index         # Dense index (built by ingest.py)
│   ├── bm25.pkl            # Sparse index
│   └── chunks.json         # Chunk text + metadata
├── src/
│   ├── ingest.py           # PDF → chunks → indexes
│   ├── retriever.py        # Dense + sparse + RRF
│   ├── reranker.py         # Cross-encoder reranking
│   ├── eval.py             # Recall@k, MRR, NDCG evaluation
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/yourusername/LegalMind
cd LegalMind
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Add your PDFs to `data/` and `data/judgements/`, then:

```bash
python src/ingest.py      # builds indexes (run once)
uvicorn api.main:app --reload
```

API available at `http://localhost:8000/docs`

---

## API

**POST `/query`**
```json
{
  "question": "What is the rarest of rare doctrine for death penalty?",
  "top_k": 5,
  "use_reranker": true
}
```

**GET `/stats`** — corpus statistics  
**GET `/health`** — server health check

---

## Limitations & Future Work

- **Reranker domain mismatch** — fine-tune cross-encoder on Indian legal query-passage pairs
- **OCR support** — scanned PDFs currently skipped; add `pytesseract` for full coverage
- **BNS 2023** — extend corpus with Bharatiya Nyaya Sanhita sections as SC case law develops
- **Chunking strategy** — replace word-based splitting with sentence-boundary-aware chunking using spaCy
- **Larger reranker** — `ms-marco-MiniLM-L-12-v2` generalises better across domains
- **LLM answer generation** — wire a generative model on top of retrieved chunks for full RAG

---

## What This Demonstrates

- Hybrid retrieval combining dense and sparse signals without score normalisation
- RRF as a scale-free rank fusion method
- Quantitative eval framework measuring four retrieval configurations
- Understanding of why rerankers fail under domain mismatch
- FastAPI serving pattern with models loaded once at startup
