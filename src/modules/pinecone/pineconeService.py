import os
from typing import Optional
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import hashlib

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX", "analyzer-index")

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

def _stable_id(namespace: str, chunk: str, i: int) -> str:
    h = hashlib.sha1(f"{namespace}|{i}|{chunk}".encode("utf-8")).hexdigest()[:20]
    return f"{namespace}-{h}"

def delete_namespace(namespace: str) -> None:
    index.delete(namespace=namespace, delete_all=True)

def delete_ids(ids: list[str], namespace: str) -> None:
    if ids:
        index.delete(ids=ids, namespace=namespace)

def upsert_chunks(
    chunks: list[str],
    vectors: list[list[float]],
    namespace: str,
    metadata: dict = {},
    batch_size: int = 100,
) -> int:
    assert len(chunks) == len(vectors), "chunks and vectors length mismatch"

    ids = [_stable_id(namespace, chunk, i) for i, chunk in enumerate(chunks)]
    payloads = [
        {"id": id_, "values": vec, "metadata": {"text": chunk, **metadata}}
        for id_, vec, chunk in zip(ids, vectors, chunks)
    ]

    total = 0
    for i in range(0, len(payloads), batch_size):
        batch = payloads[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        total += len(batch)
    return total

def query_chunks(
    query_vector: list[float],
    top_k: Optional[int] = 5,
    namespace: str = "",
    score_threshold: Optional[float] = None,
    metadata_filter: Optional[dict] = None,
) -> list[dict]:
    # safeguard top_k
    top_k = int(top_k) if top_k and int(top_k) > 0 else 5

    res = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=metadata_filter or None,
    )
    matches = [
        {
            "id": match["id"],
            "score": match["score"],
            "text": match["metadata"].get("text"),
            "metadata": match["metadata"],
        }
        for match in res["matches"]
        if score_threshold is None or match["score"] >= score_threshold
    ]
    return matches
