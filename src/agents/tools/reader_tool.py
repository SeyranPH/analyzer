import json
from typing import Dict, Any
from .utils import extract_arxiv_id, download_pdf, extract_text_from_pdf

def reader_tool(arxiv_url: str, emit=None) -> Dict[str, Any]:
    """Download and extract text from an ArXiv PDF.
    Returns: {ok: bool, arxiv_id: str, text: str | None, error: str | None}
    """
    if emit:
        emit("reader.start", {"url": arxiv_url})

    arxiv_id = extract_arxiv_id(arxiv_url)
    if not arxiv_id:
        return {"ok": False, "error": "Invalid ArXiv URL format", "url": arxiv_url}

    if emit:
        emit("reader.download.start", {"arxiv_id": arxiv_id})

    pdf_content = download_pdf(arxiv_id)
    if not pdf_content:
        return {"ok": False, "error": f"Could not download PDF for {arxiv_id}. Check server logs for details.", "arxiv_id": arxiv_id, "url": arxiv_url}

    if emit:
        emit("reader.download.end", {"arxiv_id": arxiv_id, "size_bytes": len(pdf_content)})

    if emit:
        emit("reader.extract.start", {"arxiv_id": arxiv_id})

    try:
        text = extract_text_from_pdf(pdf_content)
        if emit:
            emit("reader.extract.end", {"arxiv_id": arxiv_id, "chars": len(text)})
        return {"ok": True, "arxiv_id": arxiv_id, "text": text}
    except Exception as e:
        error_msg = f"Failed to extract text from PDF: {str(e)}"
        if emit:
            emit("reader.extract.error", {"arxiv_id": arxiv_id, "error": error_msg})
        return {"ok": False, "error": error_msg, "arxiv_id": arxiv_id}

try:
    from pydantic import BaseModel, Field
    from langchain.tools import StructuredTool

    class _ReaderArgs(BaseModel):
        arxiv_url: str = Field(..., description="ArXiv abstract URL, e.g. https://arxiv.org/abs/2410.16930")

    reader_lc_tool = StructuredTool.from_function(
        name="read_arxiv_pdf_tool",
        description="Download and extract full text from an ArXiv PDF given its abstract URL. Returns {ok, text, arxiv_id}.",
        func=reader_tool,
        args_schema=_ReaderArgs,
    )
except Exception:
    reader_lc_tool = None

reader_openai_schema = {
    "name": "read_arxiv_pdf_tool",
    "description": "Download and extract full text from an ArXiv PDF given its abstract URL. Returns {ok, text, arxiv_id}.",
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
    url = args.get("arxiv_url")
    if not url:
        return {"ok": False, "error": "Missing required parameter 'arxiv_url'"}
    return reader_tool(url)
