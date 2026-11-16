
import requests

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text:latest"


def generate_embedding(text):
    """
    Generate a 768-dimensional embedding using Ollama.
    """

    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    response = requests.post(
        OLLAMA_EMBED_URL,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    embeddings = result.get("embeddings")

    if not embeddings or not embeddings[0]:
        raise ValueError("Ollama returned an empty embedding.")

    embedding = embeddings[0]

    if len(embedding) != 768:
        raise ValueError(
            f"Invalid embedding dimension. Expected 768, "
            f"received {len(embedding)}."
        )

    return embedding
