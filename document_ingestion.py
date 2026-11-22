
from pdf_processor import extract_pdf_pages
from document_chunking import split_into_chunks, CHUNK_SIZE, CHUNK_OVERLAP
from document_metadata import generate_embedding


def process_pdf(pdf_bytes: bytes):

    # 1. Extract text page by page
    pages = extract_pdf_pages(pdf_bytes)

    document_chunks = []

    # 2. Split each page into overlapping chunks
    for page_number, page_text in pages.items():

        chunks = split_into_chunks(
            page_text,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        # 3. Generate embeddings for every chunk
        for chunk_index, chunk_text in enumerate(chunks, start=1):

            embedding = generate_embedding(chunk_text)

            document_chunks.append({
                "page_number": page_number,
                "chunk_index": chunk_index,
                "content": chunk_text,
                "embedding": embedding
            })

    return {
        "total_pages": len(pages),
        "total_chunks": len(document_chunks),
        "chunks": document_chunks
    }


if __name__ == "__main__":

    from pathlib import Path

    pdf_path = Path("sampleTest.pdf")

    result = process_pdf(pdf_path.read_bytes())

    print("PDF ingestion completed!")
    print("Total pages:", result["total_pages"])
    print("Total chunks:", result["total_chunks"])

    for chunk in result["chunks"]:
        print("\n----------------------------")
        print("Page:", chunk["page_number"])
        print("Chunk:", chunk["chunk_index"])
        print("Content:", chunk["content"])
        print("Embedding dimensions:", len(chunk["embedding"]))


