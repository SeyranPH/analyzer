from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import asyncio

from src.agents.tools import scout_tool, reader_tool, processor_tool
from src.agents.answering_agent import run_answering_agent_stream, run_answering_agent

analysisRouter = APIRouter(prefix="/analysis", tags=["analysis"])

class CreateAnalysisBody(BaseModel):
    analysisQuery: str

class ReadPdfBody(BaseModel):
    pdfUrl: str

class ProcessorText(BaseModel):
    text: str
    namespace: str = "default"
    meta: Dict[str, Any] | None = None


class AnswerRequest(BaseModel):
    question: str
    namespace: str = "default"
    threshold: float = 0.70

@analysisRouter.post("/arxiv-query", status_code=status.HTTP_201_CREATED)
async def create_analysis(body: CreateAnalysisBody):
    results = scout_tool(body.analysisQuery)
    return {"arxiv_results": results}


@analysisRouter.post("/pdf", status_code=status.HTTP_201_CREATED)
async def read_pdf(body: ReadPdfBody):
    results = reader_tool(body.pdfUrl)
    return {"pdf_text": results}


@analysisRouter.post("/processor", status_code=status.HTTP_201_CREATED)
async def process_text(body: ProcessorText):
    results = processor_tool(body.text, namespace=body.namespace, meta=body.meta)
    return {"processor_result": results}


@analysisRouter.post("/answer", status_code=status.HTTP_200_OK)
async def answer(req: AnswerRequest):
    """
    Streaming endpoint (SSE) that uses the intelligent agent to answer questions.
    """

    async def event_generator():
        steps: List[Dict[str, Any]] = []

        def emit(event: str, data: Dict[str, Any]):
            steps.append({"event": event, "data": data})

        try:
            async for result in run_answering_agent_stream(
                question=req.question,
                namespace=req.namespace,
                threshold=req.threshold,
                emit=emit
            ):
                yield f"data: {json.dumps({'type': 'result', 'data': result})}\n\n"

        except Exception as e:
            error_msg = {"event": "error", "data": {"error": str(e)}}
            yield f"data: {json.dumps(error_msg)}\n\n"

        for step in steps:
            yield f"data: {json.dumps(step)}\n\n"

        yield f"data: {json.dumps({'event': 'complete', 'data': {'steps': steps}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

@analysisRouter.post("/answer/sync", status_code=status.HTTP_200_OK)
async def answer_sync(req: AnswerRequest):
    """
    Synchronous endpoint for backward compatibility.
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
