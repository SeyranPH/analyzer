import faiss
import numpy as np
import uuid
from typing import List

class SimpleVectorStore:
    """Simple vector store using FAISS for similarity search."""
    
    def __init__(self, dim: int = 1536):
        self.index = faiss.IndexFlatL2(dim)
        self.id_to_chunk = {}
        self.chunk_ids = []

    def add(self, vectors: List[List[float]], chunks: List[str]):
        """Add vectors and their corresponding text chunks to the store."""
        if not vectors or not chunks:
            return
            
        np_vectors = np.array(vectors).astype("float32")
        self.index.add(np_vectors)
        
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            self.id_to_chunk[chunk_id] = chunk
            self.chunk_ids.append(chunk_id)

    def search(self, query_vector: List[float], k: int = 5) -> List[str]:
        """Search for similar chunks using a query vector."""
        if not query_vector or self.index.ntotal == 0:
            return []
            
        np_query = np.array([query_vector]).astype("float32")
        distances, indices = self.index.search(np_query, k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.chunk_ids):
                chunk_id = self.chunk_ids[idx]
                results.append(self.id_to_chunk[chunk_id])
        
        return results 