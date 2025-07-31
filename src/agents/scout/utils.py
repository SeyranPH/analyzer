import httpx
import feedparser
from typing import List

ARXIV_API_URL = "https://export.arxiv.org/api/query"

def search_arxiv(query: str, max_results: int = 5) -> List[dict]:
    """Search arXiv API and return parsed results."""
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }

    try:
        with httpx.Client(follow_redirects=True) as client:
            response = client.get(ARXIV_API_URL, params=params)
            response.raise_for_status()

        feed = feedparser.parse(response.text)

        results = []
        for entry in feed.entries:
            results.append({
                "title": entry.title,
                "summary": entry.summary,
                "authors": [author.name for author in entry.authors],
                "published": entry.published,
                "url": entry.link
            })

        return results
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return [] 