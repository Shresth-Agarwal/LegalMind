from fastapi import FastAPI
from .routers import router
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from src.retriever import load_indexes
from src.reranker import load_reranker

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads models when the server starts 
    keeps them in memory for the lifetime of the app
    """
    print("Loading models...")
    chunks, bm25, faiss_index, model = load_indexes()
    tokenizer, rerank_model = load_reranker()
    app.state.chunks = chunks
    app.state.bm25 = bm25
    app.state.faiss_index = faiss_index
    app.state.embed_model = model
    app.state.rerank_tokenizer = tokenizer
    app.state.rerank_model = rerank_model
    
    yield
    
app = FastAPI(
    title="LegalMind API",
    description="API for LegalMind application",
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(router)
