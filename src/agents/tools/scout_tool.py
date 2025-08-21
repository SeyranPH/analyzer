from .utils import search_arxiv

def scout_tool(query: str, max_results: int = 5):
    """Tool for searching ArXiv for papers."""
    return search_arxiv(query, max_results) 