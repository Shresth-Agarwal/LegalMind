import os
import json
import pickle
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

DATA_DIR      = Path("data")
JUDGMENTS_DIR = DATA_DIR / "judgments"
INDEX_DIR     = Path("indexes")
INDEX_DIR.mkdir(exist_ok=True)

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80   
EMBED_MODEL   = "all-MiniLM-L6-v2"

# Step 1: Extract text from a single PDF

def extract_text(pdf_path: Path) -> str:
    """
    Reads every page of a PDF and joins them into one string.
    PdfReader handles both text-layer PDFs and basic formatting.
    """
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:  # some pages are blank or image-only
            pages.append(text)
    return "\n".join(pages)


# Step 2: Split text into overlapping word-based chunks

def chunk_text(text: str, source: str) -> list[dict]:
    """
    Splits text into overlapping chunks of ~400 words each.

    Why word-based and not character-based?
    Legal text has long words. Word count maps better to
    semantic density than character count.

    Why overlap?
    A key sentence sitting at a chunk boundary would be split
    in half and missed by retrieval. Overlap ensures every
    sentence appears fully in at least one chunk.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        chunk_text  = " ".join(chunk_words)

        chunks.append({
            "text":   chunk_text,
            "source": source,           # filename — for metadata
            "chunk_id": len(chunks)     # position within this document
        })

        # Move forward by (CHUNK_SIZE - CHUNK_OVERLAP) so windows overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# Step 3: Load all documents and collect chunks

def load_all_chunks() -> list[dict]:
    all_chunks = []

    # IPC bare act
    bare_act_path = DATA_DIR / "ipc_act.pdf"
    if bare_act_path.exists():
        print("Loading IPC Bare Act...")
        text   = extract_text(bare_act_path)
        chunks = chunk_text(text, source="ipc_bare_act.pdf")
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks")
    else:
        print("WARNING: ipc_bare_act.pdf not found in data/")

    # All judgment PDFs
    print("\nLoading judgments...")
    judgment_files = list(JUDGMENTS_DIR.glob("*.pdf"))

    if not judgment_files:
        print("WARNING: No PDFs found in data/judgments/")

    for pdf_path in tqdm(judgment_files):
        text   = extract_text(pdf_path)
        chunks = chunk_text(text, source=pdf_path.name)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")
    return all_chunks


# Step 4: Build FAISS dense index

def build_faiss_index(chunks: list[dict], model: SentenceTransformer):
    """
    Embeds every chunk and stores vectors in a FAISS flat index.

    Why FlatL2 and not IVF or HNSW?
    Your corpus is small (~2,000 chunks). Exact search is fast
    enough and gives perfect recall. Approximate indexes
    (IVF/HNSW) trade recall for speed — unnecessary here.
    """
    print("\nBuilding FAISS dense index...")
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=32,
                              show_progress_bar=True,
                              convert_to_numpy=True)

    dim   = embeddings.shape[1]           # 384 for MiniLM
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    # Save index
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    print(f"  → FAISS index saved ({index.ntotal} vectors, dim={dim})")

    return index, embeddings


# Step 5: Build BM25 sparse index

def build_bm25_index(chunks: list[dict]):
    """
    Tokenizes each chunk and builds a BM25Okapi index.

    BM25 is a classical term-frequency ranking function.
    It will score chunks highly when query terms appear
    frequently in a chunk but rarely across the whole corpus —
    perfect for exact statute references like 'Section 302'.

    Why does dense search miss this?
    Embeddings compress meaning into vectors. Two different
    section numbers can end up with similar vectors if the
    surrounding legal language is similar. BM25 never conflates
    '302' with '304' because they are different tokens.
    """
    print("\nBuilding BM25 sparse index...")
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenized)

    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("  → BM25 index saved")

    return bm25


# Step 6: Save chunk metadata

def save_chunks(chunks: list[dict]):
    """
    Saves all chunk text + metadata as a JSON file.
    This is your lookup table — given a chunk index,
    you can retrieve the original text and its source document.
    """
    with open(INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  → chunks.json saved ({len(chunks)} entries)")

if __name__ == "__main__":
    # Load embedding model once — reused for FAISS index building
    print(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    chunks = load_all_chunks()
    build_faiss_index(chunks, model)
    build_bm25_index(chunks)
    save_chunks(chunks)