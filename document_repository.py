
import hashlib

from database import get_db_connection


def persist_document(file_name, pdf_bytes, chunks):
    """
    Save document metadata and all extracted chunks in PostgreSQL.

    Uses the file SHA-256 hash as a deterministic document identifier.
    Re-uploading the same file replaces its existing chunks.
    """

    if not file_name:
        raise ValueError("File name cannot be empty.")

    if not pdf_bytes:
        raise ValueError("PDF content cannot be empty.")

    if not chunks:
        raise ValueError("No document chunks available to persist.")

    # 1. Generate a deterministic document ID.
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    document_id = f"DOC-{file_hash}"

    # 2. Prepare database records.
    chunk_records = []

    for global_index, chunk in enumerate(chunks, start=1):
        content = chunk["content"]
        embedding = chunk["embedding"]
        page_number = chunk["page_number"]

        if not content or len(embedding) != 768:
            raise ValueError(
                f"Invalid content or embedding for chunk {global_index}."
            )

        chunk_id = f"{document_id}-C{global_index}"

        chunk_records.append((
            chunk_id,
            document_id,
            file_name,
            page_number,
            global_index,
            content,
            embedding
        ))

    # 3. Persist everything in one transaction.
    with get_db_connection() as connection:
        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO documents (
                    document_id,
                    file_name,
                    file_hash,
                    updated_at
                )
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (document_id)
                DO UPDATE SET
                    file_name = EXCLUDED.file_name,
                    file_hash = EXCLUDED.file_hash,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (document_id, file_name, file_hash)
            )

            # Remove old chunks when the same PDF is uploaded again.
            cursor.execute(
                """
                DELETE FROM document_chunks
                WHERE document_id = %s
                """,
                (document_id,)
            )

            # Insert chunks with document-wide unique indexes.
            cursor.executemany(
                """
                INSERT INTO document_chunks (
                    chunk_id,
                    document_id,
                    file_name,
                    page_number,
                    chunk_index,
                    content,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                chunk_records
            )

    return {
        "document_id": document_id,
        "file_name": file_name,
        "file_hash": file_hash,
        "total_chunks": len(chunk_records)
    }
