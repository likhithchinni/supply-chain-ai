
import requests
import math

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text:latest"


# Sample logistics knowledge base
documents = [
    {
        "id": 1,
        "title": "Customs Clearance Delay",
        "content": """
        International shipments can be delayed due to customs clearance.
        Missing invoices, incorrect declarations, and incomplete documents
        may cause customs authorities to hold shipments.
        """
    },
    {
        "id": 2,
        "title": "Shipment Tracking Policy",
        "content": """
        Customers can track their shipments using the tracking number.
        Tracking information includes shipment location, transit status,
        estimated delivery date, and delivery updates.
        """
    },
    {
        "id": 3,
        "title": "Missing Documentation",
        "content": """
        Commercial invoices, packing lists, and certificates of origin
        are required for international shipping. Missing documents can
        prevent customs clearance.
        """
    },
    {
        "id": 4,
        "title": "Warehouse Processing",
        "content": """
        Warehouse teams receive, inspect, scan, sort, and dispatch packages.
        Processing delays can occur due to high shipment volumes.
        """
    },
    {
        "id": 5,
        "title": "Delivery Exception",
        "content": """
        A delivery exception occurs when a package cannot be delivered
        as planned. Common reasons include incorrect addresses,
        recipient unavailability, and transportation issues.
        """
    },
    {
        "id": 6,
        "title": "Java Backend Application",
        "content": """
        Java and Spring Boot are used to develop backend applications.
        REST APIs allow applications to communicate with other services.
        """
    }
]


# Generate embedding using Ollama
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

    return response.json()["embeddings"][0]


# Calculate cosine similarity
def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# Prepare document embeddings
def prepare_knowledge_base():
    knowledge_base = []

    print("Generating document embeddings...")

    for document in documents:
        text = document["title"] + "\n" + document["content"]

        embedding = generate_embedding(text)

        knowledge_base.append({
            "id": document["id"],
            "title": document["title"],
            "content": document["content"],
            "embedding": embedding
        })

    print("Knowledge base is ready!\n")

    return knowledge_base


# Search documents using semantic similarity
def semantic_search(query, knowledge_base, top_k=3):
    query_embedding = generate_embedding(query)

    results = []

    for document in knowledge_base:
        score = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        results.append({
            "title": document["title"],
            "content": document["content"],
            "score": score
        })

    # Sort by similarity in descending order
    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]


# Main execution
if __name__ == "__main__":

    knowledge_base = prepare_knowledge_base()

    query = "Why is my shipment delayed?"
    # query = "What documents are needed for international shipping?"

    print("User Query:", query)
    print("\nTop 3 Search Results:\n")

    results = semantic_search(
        query,
        knowledge_base,
        top_k=3
    )

    for index, result in enumerate(results, start=1):
        print(f"Rank: {index}")
        print(f"Document: {result['title']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Content: {result['content'].strip()}")
        print("-" * 60)
