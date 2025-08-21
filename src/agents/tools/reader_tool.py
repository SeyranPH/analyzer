from typing import Optional, Dict, Any
import json
from .utils import extract_arxiv_id, download_pdf, extract_text_from_pdf

def reader_tool(arxiv_url: str) -> Dict[str, Any]:
    """Tool for downloading and extracting text from ArXiv PDFs."""
    arxiv_id = extract_arxiv_id(arxiv_url)
    if not arxiv_id:
        return {"ok": False, "error": "Invalid ArXiv URL format"}

    pdf_content = download_pdf(arxiv_id)
    if not pdf_content:
        return {"ok": False, "error": "Could not download PDF", "arxiv_id": arxiv_id}

    text = extract_text_from_pdf(pdf_content)
    return {"ok": True, "arxiv_id": arxiv_id, "text": text}

try:
    from pydantic import BaseModel, Field
    from langchain.tools import StructuredTool

    class _ReaderArgs(BaseModel):
        arxiv_url: str = Field(..., description="ArXiv abstract URL, e.g. https://arxiv.org/abs/2410.16930")

    reader_lc_tool = StructuredTool.from_function(
        name="read_arxiv_pdf_tool",
        description="Download and extract full text from an ArXiv PDF given its abstract URL.",
        func=lambda arxiv_url: reader_tool(arxiv_url),
        args_schema=_ReaderArgs,
    )
except Exception:
    reader_lc_tool = None

reader_openai_schema = {
    "name": "read_arxiv_pdf_tool",
    "description": "Download and extract full text from an ArXiv PDF given its abstract URL.",
    "parameters": {
        "type": "object",
        "properties": {
            "arxiv_url": {"type": "string", "description": "ArXiv abstract URL"}
        },
        "required": ["arxiv_url"]
    }
}

def execute_reader_openai_tool(arguments_json: str) -> Dict[str, Any]:
    args = json.loads(arguments_json or "{}")
    return reader_tool(args["arxiv_url"])
