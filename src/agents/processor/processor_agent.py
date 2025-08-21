from .chunker import split_text
from .embedder import embed_texts
from .vector_store import SimpleVectorStore

def process_and_search(text: str, query: str, chunk_size: int = 500, overlap: int = 50, k: int = 5) -> list[str]:
    """
    Process text through the complete pipeline: chunk → embed → store → search.
    
    Args:
        text: The text to process
        query: The search query
        chunk_size: Size of text chunks
        overlap: Overlap between chunks
        k: Number of results to return
    
    Returns:
        List of matching text chunks
    """
    # Step 1: Split text into chunks
    chunks = split_text(text, chunk_size, overlap)
    
    # Step 2: Generate embeddings for chunks
    chunk_embeddings = embed_texts(chunks)
    
    # Step 3: Create vector store and add chunks
    store = SimpleVectorStore()
    store.add(chunk_embeddings, chunks)
    
    # Step 4: Generate embedding for query
    query_embedding = embed_texts([query])
    
    # Step 5: Search for similar chunks
    if query_embedding:
        matches = store.search(query_embedding[0], k)
        return matches
    
    return [] 