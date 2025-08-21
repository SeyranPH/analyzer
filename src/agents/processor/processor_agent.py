from src.agents.processor.utils import split_text
from src.modules.openai.openaiService import get_embeddings
from src.modules.pinecone.pineconeService import upsert_chunks


def processor_agent(text: str):
    """
    Process text, get embeddings, and upsert to Pinecone.
    """
    chunks = split_text(text)
    vectors = get_embeddings(chunks)
    upsert_chunks(chunks, vectors, "default")
    return chunks
