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
    text = re.sub(r'\s+', ' ', text.strip())
    chunks = []
    start = 0
    
    while start < len(text):
        ideal_end = start + chunk_size
        if ideal_end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break
        split_point = find_best_split_point(text, start, ideal_end, chunk_size)
        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, split_point - overlap)
    
    return chunks

def find_best_split_point(text: str, start: int, ideal_end: int, chunk_size: int) -> int:
    """
    Find the best split point within the chunk, preferring:
    1. Paragraph breaks (double newlines)
    2. Sentence endings (. ! ?)
    3. Word boundaries
    4. Fallback to character boundary
    """
    search_start = max(start, ideal_end - chunk_size // 4)
    search_end = min(len(text), ideal_end + chunk_size // 4)
    search_text = text[search_start:search_end]
    
    para_patterns = [
        r'\n\s*\n',  # Double newlines
        r'\n\s*[-=*]{3,}\s*\n',  # Section dividers
        r'\n\s*\d+\.\s*\n',  # Numbered sections
    ]
    
    for pattern in para_patterns:
        matches = list(re.finditer(pattern, search_text))
        if matches:
            best_match = None
            best_distance = float('inf')
            for match in matches:
                distance = abs((search_start + match.end()) - ideal_end)
                if distance < best_distance:
                    best_distance = distance
                    best_match = match
            
            if best_match:
                return search_start + best_match.end()
    
    sentence_pattern = r'[.!?]+\s+'
    matches = list(re.finditer(sentence_pattern, search_text))
    if matches:
        best_match = None
        best_distance = float('inf')
        for match in matches:
            distance = abs((search_start + match.end()) - ideal_end)
            if distance < best_distance:
                best_distance = distance
                best_match = match
        
        if best_match:
            return search_start + best_match.end()
    
    word_pattern = r'\s+'
    matches = list(re.finditer(word_pattern, search_text))
    if matches:
        # Find the match closest to ideal_end
        best_match = None
        best_distance = float('inf')
        for match in matches:
            distance = abs((search_start + match.end()) - ideal_end)
            if distance < best_distance:
                best_distance = distance
                best_match = match
        
        if best_match:
            return search_start + best_match.end()

    return ideal_end

def extract_arxiv_id(arxiv_url: str) -> Optional[str]:
    """Extract ArXiv ID from URL, handling version numbers properly."""
    match = re.search(r'arxiv\.org/abs/([^/]+(?:/[^/]+)?)(?:v\d+)?', arxiv_url)
    return match.group(1) if match else None

def download_pdf(arxiv_id: str) -> Optional[bytes]:
    """Download PDF content from ArXiv."""
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            response = client.get(pdf_url)
            response.raise_for_status()
            return response.content
    except httpx.HTTPStatusError as e:
        print(f"HTTP error downloading PDF {arxiv_id}: {e.response.status_code} - {e.response.text[:200]}")
        return None
    except httpx.TimeoutException as e:
        print(f"Timeout downloading PDF {arxiv_id}: {e}")
        return None
    except httpx.RequestError as e:
        print(f"Request error downloading PDF {arxiv_id}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error downloading PDF {arxiv_id}: {e}")
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
        pass 
        return []
    except Exception as e:
        pass 
        return [] 