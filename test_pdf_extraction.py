
from pathlib import Path

from pdf_processor import extract_pdf_pages

pdf_path = Path("sampleTest.pdf")

if not pdf_path.exists():
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

pdf_bytes = pdf_path.read_bytes()

pages = extract_pdf_pages(pdf_bytes)

print("PDF extraction successful!")
print("Total pages:", len(pages))

for page_number, text in pages.items():
    print(f"\n--- Page {page_number} ---")
    print("Extracted characters:", len(text))
    print("Extracted text:")
    print(text[:500])


