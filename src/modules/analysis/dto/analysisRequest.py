from pydantic import BaseModel

class AnalysisRequest(BaseModel):
    arxiv_url: str
    query: str
    chunk_size: int = 500
    overlap: int = 50
    k: int = 5
