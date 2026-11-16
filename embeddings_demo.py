import requests
import math

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"


def generate_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "input": text
        },
        timeout=60
    )

    response.raise_for_status()

    data = response.json()
    return data["embeddings"][0]


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


sentences = [
    "Shipment is delayed due to customs clearance.",
    "The package delivery is running late.",
    "Java is a programming language."
]

embeddings = [generate_embedding(text) for text in sentences]

print("Embedding dimensions:", len(embeddings[0]))

print("\nSimilarity scores:")

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = cosine_similarity(embeddings[i], embeddings[j])

        print(f"\nSentence 1: {sentences[i]}")
        print(f"Sentence 2: {sentences[j]}")
        print(f"Similarity: {score:.4f}")