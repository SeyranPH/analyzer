from .utils import extract_arxiv_id, download_pdf, extract_text_from_pdf

def reader_tool(arxiv_url: str) -> str:
    """Tool for downloading and extracting text from ArXiv PDFs."""
    arxiv_id = extract_arxiv_id(arxiv_url)
    if not arxiv_id:
        return "Error: Invalid ArXiv URL format"
    
    pdf_content = download_pdf(arxiv_id)
    if not pdf_content:
        return "Error: Could not download PDF"
    
    return extract_text_from_pdf(pdf_content) 