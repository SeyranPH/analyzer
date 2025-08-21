from .utils import search_arxiv

def scout_agent(query: str, max_results: int = 5):
    """Main scout agent function that searches ArXiv for papers."""
    return search_arxiv(query, max_results) 