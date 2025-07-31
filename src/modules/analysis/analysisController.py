from fastapi import APIRouter, status
from src.modules.analysis.dto.analysisRequest import AnalysisRequest
from src.agents.scout import scout_agent
from src.agents.reader import reader_agent
from src.agents.processor import process_and_search

analysisRouter = APIRouter(prefix="/analysis", tags=["analysis"])

@analysisRouter.post("/", status_code=status.HTTP_201_CREATED)
async def create_analysis(analysisQuery: str):
    results = scout_agent(analysisQuery)
    return {"arxiv_results": results}

@analysisRouter.post("/pdf", status_code=status.HTTP_201_CREATED)
async def read_pdf(pdfUrl: str):
    results = reader_agent(pdfUrl)
    return {"pdf_text": results}

@analysisRouter.post("/analyze", status_code=status.HTTP_201_CREATED)
async def analyze_pdf(request: AnalysisRequest):
    """Analyze PDF content using semantic search."""
    text = reader_agent(request.arxiv_url)
    
    if text.startswith("Error"):
        return {"error": text}
    
    matches = process_and_search(
        text=text,
        query=request.query,
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        k=request.k
    )
    
    return {
        "query": request.query,
        "arxiv_url": request.arxiv_url,
        "matches": matches,
        "total_matches": len(matches)
    }