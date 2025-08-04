import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX", "analyzer-index")

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

def upsert_chunks(chunks: list[str], vectors: list[list[float]], namespace: str, metadata: dict = {}):
    """
    Upsert chunks + vectors to Pinecone in a given namespace.
    """
    ids = [f"{namespace}-{i}" for i in range(len(chunks))]
    
    vectors_to_upsert = []
    for id_, vec, chunk in zip(ids, vectors, chunks):
        vectors_to_upsert.append({
            "id": id_,
            "values": vec,
            "metadata": {
                "text": chunk,
                **metadata
            }
        })

    index.upsert(vectors=vectors_to_upsert, namespace=namespace)


def query_chunks(query_vector: list[float], top_k: int = 5, namespace: str = "") -> list[str]:
    """
    Query Pinecone and return text chunks from metadata.
    """
    res = index.query(
        vector=query_vector, 
        top_k=top_k, 
        namespace=namespace, 
        include_metadata=True
    )
    return [match['metadata']['text'] for match in res['matches']] 