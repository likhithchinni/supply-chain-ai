
import requests
from datetime import datetime, timezone

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text:latest"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10

DOCUMENT_ID = "DOC-101"
FILE_NAME = "customs_clearance_policy.pdf"

# Move your existing two-page dictionary here.
PAGES = {
    1: """
    International shipments require proper documentation before
    customs clearance can be completed. Commercial invoices,
    packing lists, and certificates of origin are important
    documents. Customs authorities verify shipment information
    against the submitted documents. Incorrect invoice values,
    missing certificates, and incomplete declarations can result
    in shipment delays. Logistics teams should verify all documents
    before dispatch. Customs officers may hold shipments until
    missing information is provided.
    """,

    2: """
    Warehouse teams receive, inspect, scan, sort, and dispatch
    packages. Processing delays can occur due to high shipment
    volumes. Every package must have the correct shipping label.
    Shipment tracking information should be updated after every
    important movement. Delivery exceptions must be reported
    immediately. Transportation teams should coordinate with
    warehouse teams to resolve delivery problems.
    """
}

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


def split_into_chunks(text):
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE

        chunk_text = " ".join(words[start:end])
        chunks.append(chunk_text)

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def create_document_chunks(document_id, file_name, pages):

    document_chunks = []

    chunk_counter = 1

    current_time = datetime.now(timezone.utc).isoformat()

    for page_number, page_text in pages.items():

        chunks = split_into_chunks(page_text)

        for chunk in chunks:

            chunk_id = (
                f"{document_id}-CH-{chunk_counter:03d}"
            )

            embedding = generate_embedding(chunk)

            chunk_record = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "file_name": file_name,
                "page_number": page_number,
                "chunk_index": chunk_counter - 1,
                "content": chunk,
                "created_at": current_time,
                "updated_at": current_time,
                "embedding": embedding
            }

            document_chunks.append(chunk_record)

            chunk_counter += 1

    return document_chunks


if __name__ == "__main__":
    print("Processing document...")

    document_chunks = create_document_chunks(
        DOCUMENT_ID,
        FILE_NAME,
        PAGES
    )

    print("\nTotal chunks:", len(document_chunks))

    for record in document_chunks:
        print("\n----------------------------------")
        print("Chunk ID:", record["chunk_id"])
        print("Page Number:", record["page_number"])
        print("Chunk Index:", record["chunk_index"])
        print("Embedding Dimensions:", len(record["embedding"]))

    print("\nDocument metadata processing completed!")