
import requests

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text:latest"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10


def split_into_chunks(text, chunk_size, overlap):

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size"
        )

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk_words = words[start:end]

        chunk_text = " ".join(chunk_words)

        chunks.append(chunk_text)

        if end >= len(words):
            break

        start = end - overlap

    return chunks

def verify_overlap(chunks, overlap):

    for i in range(len(chunks) - 1):

        current_words = chunks[i].split()
        next_words = chunks[i + 1].split()

        current_end = current_words[-overlap:]
        next_start = next_words[:overlap]

        print(f"\nChunks {i + 1} and {i + 2}")

        print("Overlap verified:", current_end == next_start)


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


if __name__ == "__main__":

    document = """
    International shipments require proper documentation before customs
    clearance can be completed. Commercial invoices, packing lists,
    and certificates of origin are important documents.

    Customs authorities verify shipment information against the submitted
    documents. Incorrect invoice values, missing certificates, and
    incomplete declarations can result in shipment delays.

    When customs authorities identify documentation problems, they may
    hold the shipment until the required information is provided.

    Logistics teams should verify all shipping documents before dispatch.
    They should check invoice details, consignee information, shipment
    descriptions, and applicable customs requirements.

    Warehouse teams must also ensure packages are scanned, sorted,
    and dispatched according to the shipping schedule.
    """

    print("Original document word count:", len(document.split()))

    chunks = split_into_chunks(
        document,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    print("\nTotal chunks:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):

        print(f"\n--- Chunk {index} ---")
        print("Word count:", len(chunk.split()))
        print(chunk)

        embedding = generate_embedding(chunk)

        print("Embedding dimensions:", len(embedding))

    print("\nDocument chunking and embedding completed!")

    # Verify overlap only when running this file directly
    verify_overlap(chunks, CHUNK_OVERLAP)