from .utils import split_text
from src.modules.openai.openaiService import get_embeddings
from src.modules.pinecone.pineconeService import upsert_chunks

def processor_tool(text: str):
    """Tool for processing text, getting embeddings, and upserting to Pinecone."""
    chunks = split_text(text)
    vectors = get_embeddings(chunks)
    upsert_chunks(chunks, vectors, "default")
    return chunks 