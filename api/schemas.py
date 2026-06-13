from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_reranker: bool = True

class ChunkResult(BaseModel):
    rank: int
    source: str
    text: str
    reranker_score: float | None = None
    
class QueryResponse(BaseModel):
    question: str
    mode: str
    results: list[ChunkResult]