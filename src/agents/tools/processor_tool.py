from typing import Optional, Dict, Any
import json
import time

from .utils import split_text
from src.modules.openai.openaiService import get_embeddings
from src.modules.pinecone.pineconeService import upsert_chunks

def processor_tool(text: str, namespace: str = "default", meta: Optional[Dict] = None) -> Dict[str, Any]:
    try:
        chunks = split_text(text, chunk_size=500, overlap=50)
        vectors = get_embeddings(chunks)
        upserted = upsert_chunks(chunks, vectors, namespace, metadata=meta or {})
        return {"ok": True, "namespace": namespace, "chunks_count": len(chunks), "upserted": upserted}
    except Exception as e:
        return {"ok": False, "error": str(e)}

try:
    from pydantic import BaseModel, Field
    from langchain.tools import StructuredTool

    class _ProcessorArgs(BaseModel):
        text: str = Field(..., description="Raw document text to index")
        namespace: str = Field("default", description="Pinecone namespace to store vectors")
        meta: Optional[Dict] = Field(None, description="Optional metadata to attach to each chunk")

    processor_lc_tool = StructuredTool.from_function(
        name="process_and_upsert_tool",
        description="Split text, create OpenAI embeddings, and upsert to Pinecone under a namespace.",
        func=processor_tool,
        args_schema=_ProcessorArgs,
    )
except Exception:
    processor_lc_tool = None

processor_openai_schema = {
    "name": "process_and_upsert_tool",
    "description": "Split text, embed with OpenAI, and upsert into Pinecone under a namespace.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Raw document text to index"},
            "namespace": {"type": "string", "description": "Pinecone namespace", "default": "default"},
            "meta": {"type": "object", "description": "Optional metadata to attach to each chunk"}
        },
        "required": ["text"]
    }
}

def execute_processor_openai_tool(arguments_json: str) -> Dict[str, Any]:
    args = json.loads(arguments_json or "{}")
    text = args.get("text")
    if not text:
        return {"ok": False, "error": "Missing required parameter 'text'"}
    return processor_tool(
        text=text,
        namespace=args.get("namespace", "default"),
        meta=args.get("meta")
    )
