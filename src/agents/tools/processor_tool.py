from typing import Optional, Dict
import json

from .utils import split_text
from src.modules.openai.openaiService import get_embeddings
from src.modules.pinecone.pineconeService import upsert_chunks

def processor_tool(text: str, namespace: str = "default", meta: Optional[Dict] = None):
    """
    Split -> embed -> upsert into Pinecone.
    Returns a small summary payload for logging/agent use.
    """
    chunks = split_text(text)
    vectors = get_embeddings(chunks)
    upserted = upsert_chunks(chunks, vectors, namespace, metadata=meta or {})
    return {
        "namespace": namespace,
        "chunks_count": len(chunks),
        "upserted": upserted
    }

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
        func=lambda text, namespace="default", meta=None: processor_tool(text, namespace, meta),
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

def execute_processor_openai_tool(arguments_json: str):
    """
    Helper for OpenAI tool-calls:
    - parse arguments json -> call processor_tool -> return dict result
    """
    args = json.loads(arguments_json or "{}")
    return processor_tool(
        text=args["text"],
        namespace=args.get("namespace", "default"),
        meta=args.get("meta")
    )
