from .schemas import BaseModel, QueryRequest, QueryResponse, ChunkResult
from fastapi import APIRouter, HTTPException, Request
from src.retriever import retrieve
from src.reranker import rerank

router = APIRouter(
    prefix = "/RAG",
    tags = ["RAG"]
)

@router.get("/health")
def health(request: Request):
    return {"status": "ok", "chunks_loaded": len(request.app.state.chunks)}

@router.get("/stats")
def stats(request: Request):
    chunks = request.app.state.chunks
    sources = list(set(chunk["source"] for chunk in chunks))
    return {
        "total_chunks": len(chunks),
        "unique_sources": len(sources),
        "sources": sources
    }

@router.post("/query", response_model=QueryResponse)
def query(request: Request, req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    candidates = retrieve(
        req.question,
        request.app.state.chunks,
        request.app.state.bm25,
        request.app.state.faiss_index,
        request.app.state.embed_model,
        top_k=20
    )
    
    if req.use_reranker:
        final = rerank(
            req.question,
            candidates,
            request.app.state.rerank_tokenizer,
            request.app.state.rerank_model,
            top_n=req.top_k
        )
        mode = "hybrid + rerank"
    else:
        final = candidates[:req.top_k]
        mode = "hybrid"
    
    results = []
    for i, chunk in enumerate(final, 1):
        results.append(ChunkResult(
            rank=i,
            source=chunk["source"],
            text=chunk["text"][:500],
            reranker_score=chunk.get("rerank_score")
        ))
    return QueryResponse(    
        question=req.question,
        mode=mode,
        results=results
    )
    