
from io import BytesIO

from pypdf import PdfReader


def extract_pdf_pages(pdf_bytes: bytes) -> dict[int, str]:
    """
    Extract text from each PDF page.

    Returns:
        {
            1: "Text from page 1",
            2: "Text from page 2"
        }
    """

    if not pdf_bytes:
        raise ValueError("PDF content cannot be empty.")

    reader = PdfReader(BytesIO(pdf_bytes))

    if reader.is_encrypted:
        raise ValueError("Encrypted PDF files are not supported.")

    pages = {}

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text() or ""

        page_text = page_text.strip()

        if page_text:
            pages[page_number] = page_text

    if not pages:
        raise ValueError(
            "No extractable text found in PDF. "
            "The PDF may be scanned or image-based."
        )

    return pages


if __name__ == "__main__":

    print("PDF processor module loaded successfully.")
