
from fastapi import APIRouter, status
from pydantic import BaseModel
from typing import List, Dict, Any
from src.agents.tools import scout_tool, reader_tool, processor_tool
from src.agents.answering_agent import run_answering_agent

analysisRouter = APIRouter(prefix="/analysis", tags=["analysis"])

class CreateAnalysisBody(BaseModel):
    analysisQuery: str

@analysisRouter.post("/", status_code=status.HTTP_201_CREATED)
async def create_analysis(body: CreateAnalysisBody):
    results = scout_tool(body.analysisQuery)
    return {"arxiv_results": results}

class ReadPdfBody(BaseModel):
    pdfUrl: str

@analysisRouter.post("/pdf", status_code=status.HTTP_201_CREATED)
async def read_pdf(body: ReadPdfBody):
    results = reader_tool(body.pdfUrl)  # returns dict if you adopted the wrapper; ok either way
    return {"pdf_text": results}

class ProcessorText(BaseModel):
    text: str
    namespace: str = "default"
    meta: Dict[str, Any] | None = None

@analysisRouter.post("/processor", status_code=status.HTTP_201_CREATED)
async def process_text(body: ProcessorText):
    results = processor_tool(body.text, namespace=body.namespace, meta=body.meta)
    return {"processor_result": results}

class AnswerRequest(BaseModel):
    question: str
    namespace: str = "default"
    threshold: float = 0.70

@analysisRouter.post("/answer", status_code=status.HTTP_200_OK)
async def answer(req: AnswerRequest):
    """
    1) Try RAG using Pinecone.
    2) If not enough context → the agent uses tools (scout/reader/processor) to fetch & index.
    3) RAG again and return final answer + the steps taken.
    """
    steps: List[Dict[str, Any]] = []

    def emit(event: str, data: Dict[str, Any]):
        steps.append({"event": event, "data": data})

    result = run_answering_agent(
        question=req.question,
        namespace=req.namespace,
        threshold=req.threshold,
        emit=emit,
    )

    return {
        "ok": result.get("ok", True),
        "source": result.get("source"),
        "answer": result.get("answer", None),
        "matches": result.get("matches", []),
        "steps": steps,
    }
