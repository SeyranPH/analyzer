from typing import List, Dict, Any
import json
from .utils import search_arxiv

def scout_tool(query: str, max_results: int = 5, emit=None) -> List[Dict[str, Any]]:
    """Search arXiv for papers and return a list of dicts (title, url, authors, summary, published)."""
    if emit:
        emit("scout.start", {"query": query, "max_results": max_results})
    results = search_arxiv(query, max_results)
    if emit:
        emit("scout.end", {"results_count": len(results)})
    return results

try:
    from pydantic import BaseModel, Field
    from langchain.tools import StructuredTool

    class _ScoutArgs(BaseModel):
        query: str = Field(..., description="ArXiv search query")
        max_results: int = Field(5, description="Number of results to return")

    scout_lc_tool = StructuredTool.from_function(
        name="search_arxiv_tool",
        description="Search arXiv for papers and return a list of dicts with title, url, authors, summary, and published date.",
        func=scout_tool,
        args_schema=_ScoutArgs,
    )
except Exception:
    scout_lc_tool = None

scout_openai_schema = {
    "name": "search_arxiv_tool",
    "description": "Search arXiv for papers and return a list of results (title, url, authors, summary, published date).",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "description": "Number of results", "default": 5}
        },
        "required": ["query"]
    }
}

def execute_scout_openai_tool(arguments_json: str):
    args = json.loads(arguments_json or "{}")
    query = args.get("query")
    if not query:
        return {"error": "Missing required parameter 'query'"}
    return scout_tool(query, args.get("max_results", 5))
