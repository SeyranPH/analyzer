from fastapi import APIRouter, status
from src.agents.tools import scout_tool, reader_tool, processor_tool
from pydantic import BaseModel

analysisRouter = APIRouter(prefix="/analysis", tags=["analysis"])

@analysisRouter.post("/", status_code=status.HTTP_201_CREATED)
async def create_analysis(analysisQuery: str):
    results = scout_tool(analysisQuery)
    return {"arxiv_results": results}

@analysisRouter.post("/pdf", status_code=status.HTTP_201_CREATED)
async def read_pdf(pdfUrl: str):
    results = reader_tool(pdfUrl)
    return {"pdf_text": results}

class processorText(BaseModel):
    text: str

@analysisRouter.post("/processor", status_code=status.HTTP_201_CREATED)
async def process_text(processorTextInput: processorText):
    results = processor_tool(processorTextInput.text)
    return {"text_chunks": results}