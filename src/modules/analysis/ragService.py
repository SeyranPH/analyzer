from typing import List, Optional, Callable, Dict, Any
from src.modules.openai.openaiService import get_embeddings, chat_completion
from src.modules.pinecone.pineconeService import query_chunks

SYSTEM = (
    "You are a precise research assistant. Use ONLY the provided context. "
    "If the answer is not in the context, say you don't know. Cite titles/sections if present."
)

def _build_context(matches: List[dict]) -> str:
    lines = []
    for match in matches:
        metadata = match.get("metadata") or {}
        title = metadata.get("title")
        section = metadata.get("section")
        cite = " | ".join([p for p in [title, section] if p])
        header = f"[score={match['score']:.3f}] {cite}" if cite else f"[score={match['score']:.3f}]"
        lines.append(f"{header}\n{match.get('text','')}\n")
    return "\n---\n".join(lines)

def answer_with_rag(
    question: str,
    namespace: str,
    top_k: int = 6,
    threshold: float = 0.70,
    emit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> dict:
    if emit: emit("rag.embed_query.start", {"question": question})
    qvec = get_embeddings([question])[0]
    if emit: emit("rag.embed_query.end", {"dim": len(qvec)})

    if emit: emit("rag.query_pinecone.start", {"namespace": namespace, "top_k": top_k, "threshold": threshold})
    matches = query_chunks(qvec, top_k=top_k, namespace=namespace, score_threshold=threshold)
    if emit: emit("rag.query_pinecone.end", {"hits": len(matches)})

    if not matches:
        return {"ok": False, "reason": "no_context", "matches": []}

    context = _build_context(matches)
    prompt = (
        f"{SYSTEM}\n\nContext:\n{context}\n\n"
        f"Question: {question}\n\nAnswer with citations."
    )
    if emit: emit("rag.llm.answer.start", {"model": "gpt-4o-mini"})
    answer = chat_completion(prompt, max_tokens=500)
    if emit: emit("rag.llm.answer.end", {"chars": len(answer)})

    return {"ok": True, "answer": answer, "matches": matches}
