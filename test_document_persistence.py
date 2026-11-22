from pathlib import Path

from document_ingestion import process_pdf
from document_repository import persist_document


# Read the sample PDF
pdf_path = Path("sampleTest.pdf")
pdf_bytes = pdf_path.read_bytes()

print("Processing PDF...")

# Extract text, generate chunks and embeddings
result = process_pdf(pdf_bytes)

print("Pages:", result["total_pages"])
print("Chunks:", result["total_chunks"])

# Save document and chunks to PostgreSQL
saved_document = persist_document(
    file_name=pdf_path.name,
    pdf_bytes=pdf_bytes,
    chunks=result["chunks"]
)

print("\nDocument saved successfully!")
print("Document ID:", saved_document["document_id"])
print("File name:", saved_document["file_name"])
print("Total chunks saved:", saved_document["total_chunks"])