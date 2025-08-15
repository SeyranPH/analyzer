import fitz
import os
import httpx
import re
import feedparser
from tempfile import NamedTemporaryFile
from typing import Optional, List

# ArXiv API configuration
ARXIV_API_URL = "https://export.arxiv.org/api/query"

def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into chunks with specified size and overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def extract_arxiv_id(arxiv_url: str) -> Optional[str]:
    """Extract ArXiv ID from URL, handling version numbers properly."""
    match = re.search(r'arxiv\.org/abs/(\d+\.\d+)(?:v\d+)?', arxiv_url)
    return match.group(1) if match else None

def download_pdf(arxiv_id: str) -> Optional[bytes]:
    """Download PDF content from ArXiv."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        with httpx.Client(follow_redirects=True) as client:
            response = client.get(pdf_url)
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text content from PDF bytes using PyMuPDF."""
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        tmp_pdf.write(pdf_content)
        tmp_pdf_path = tmp_pdf.name

    try:
        doc = fitz.open(tmp_pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text
    finally:
        os.remove(tmp_pdf_path)

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