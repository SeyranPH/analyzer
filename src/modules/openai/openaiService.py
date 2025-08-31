import os
import time
from typing import Iterable
from functools import wraps
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rate_limit(calls_per_minute=15):
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

def get_embeddings(texts: list[str], model="text-embedding-3-small") -> list[list[float]]:
    """
    Call OpenAI embedding API and return list of vectors.
    """
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

@rate_limit(calls_per_minute=15)
def chat_completion(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 1000) -> str:
    """Simple non-streaming chat with token limits."""
    estimated_tokens = len(prompt) // 4
    if estimated_tokens > 6000:
        prompt = prompt[:24000]
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
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
