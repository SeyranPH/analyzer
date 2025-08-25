import os
from typing import Iterable
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embeddings(texts: list[str], model="text-embedding-3-small") -> list[list[float]]:
    """
    Call OpenAI embedding API and return list of vectors.
    """
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

def chat_completion(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> str:
    """Simple non-streaming chat for RAG answers."""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content

def stream_chat(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0) -> Iterable[str]:
    """
    Streaming generator of tokens. Yields text deltas as they arrive.
    You can forward these chunks to SSE/WebSocket.
    """
    with client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True,
    ) as stream:
        for event in stream:
            delta = event.choices[0].delta.content or ""
            if delta:
                yield delta