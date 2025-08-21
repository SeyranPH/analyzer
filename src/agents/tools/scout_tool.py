from typing import List, Dict, Any
import json
from .utils import search_arxiv

def scout_tool(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Tool for searching ArXiv for papers."""
    return search_arxiv(query, max_results)

try:
    from pydantic import BaseModel, Field
    from langchain.tools import StructuredTool

    class _ScoutArgs(BaseModel):
        query: str = Field(..., description="ArXiv search query")
        max_results: int = Field(5, description="Number of results to return")

    scout_lc_tool = StructuredTool.from_function(
        name="search_arxiv_tool",
        description="Search arXiv and return candidate papers (title, url, authors, published...).",
        func=lambda query, max_results=5: scout_tool(query, max_results),
        args_schema=_ScoutArgs,
    )
except Exception:
    scout_lc_tool = None

scout_openai_schema = {
    "name": "search_arxiv_tool",
    "description": "Search arXiv for papers and return a list of results.",
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
    return scout_tool(args["query"], args.get("max_results", 5))
