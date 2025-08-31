from typing import List, Optional, Callable, Dict, Any
from src.modules.openai.openaiService import get_embeddings, chat_completion
from src.modules.pinecone.pineconeService import query_chunks

SYSTEM = (
    "You are a precise research assistant. Use ONLY the provided context. "
    "If the answer is not in the context, say you don't know. Cite titles/sections if present."
)

CTX_BUDGET = 4000

def _truncate(text: str, max_chars: int) -> str:
    return text[:max_chars]

def _build_context(matches: List[dict]) -> str:
    lines = []
    used_chars = 0
    for match in matches:
        metadata = match.get("metadata") or {}
        title = metadata.get("title")
        section = metadata.get("section")
        cite = " | ".join([p for p in [title, section] if p])
        header = f"[score={match['score']:.3f}] {cite}" if cite else f"[score={match['score']:.3f}]"

        chunk = match.get("text", "")
        if used_chars + len(chunk) > CTX_BUDGET:
            chunk = _truncate(chunk, CTX_BUDGET - used_chars)
        lines.append(f"{header}\n{chunk}\n")
        used_chars += len(chunk)
        if used_chars >= CTX_BUDGET:
            break
    return "\n---\n".join(lines)


def answer_with_rag(
    question: str,
    namespace: str,
    top_k: Optional[int] = 3,
    threshold: float = 0.50,
    emit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> dict:
    top_k = int(top_k) if top_k and int(top_k) > 0 else 6

    if emit: emit("rag.embed_query.start", {"question": question})
    qvec = get_embeddings([question])[0]
    if emit: emit("rag.embed_query.end", {"dim": len(qvec)})

    if emit: emit("rag.query_pinecone.start", {"namespace": namespace, "top_k": top_k, "threshold": threshold})
    matches = query_chunks(qvec, top_k=top_k, namespace=namespace, score_threshold=threshold)
    if emit: emit("rag.query_pinecone.end", {"hits": len(matches)})

    if emit and matches:
        emit("rag.debug", {"matches": [{"score": m["score"], "text_preview": m["text"][:100]} for m in matches]})

    if not matches:
        if emit: emit("rag.retry_lower_threshold", {"new_threshold": 0.3})
        matches = query_chunks(qvec, top_k=top_k, namespace=namespace, score_threshold=0.3)
        if emit: emit("rag.retry_result", {"hits": len(matches)})
        
        if not matches:
            if emit: emit("rag.debug_no_threshold", {"checking_all_results": True})
            all_matches = query_chunks(qvec, top_k=10, namespace=namespace, score_threshold=None)
            if emit: emit("rag.debug_all_results", {
                "total_available": len(all_matches),
                "scores": [m["score"] for m in all_matches[:5]],
                "sample_texts": [m["text"][:50] for m in all_matches[:3]]
            })
            return {"ok": False, "reason": "no_context", "matches": []}

    context = _build_context(matches)
    prompt = (
        f"{SYSTEM}\n\nContext:\n{context}\n\n"
        f"Question: {question}\n\nAnswer with citations."
    )

    model = "gpt-4o-mini"
    if emit: emit("rag.llm.answer.start", {"model": model})
    answer = chat_completion(prompt, model=model, max_tokens=500)
    if emit: emit("rag.llm.answer.end", {"chars": len(answer)})

    if emit: emit("rag.complete", {"matches": len(matches), "answer_chars": len(answer)})

    return {"ok": True, "answer": answer, "matches": matches}
