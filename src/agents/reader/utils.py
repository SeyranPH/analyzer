import fitz
import os
import httpx
import re
from tempfile import NamedTemporaryFile
from typing import Optional

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