import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_embeddings(texts: list[str], model="text-embedding-3-small") -> list[list[float]]:
    """
    Call OpenAI embedding API and return list of vectors.
    """
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]