
import hashlib
import getpass
from datetime import datetime, timezone

import psycopg
from pgvector.psycopg import register_vector

from document_metadata import (
    DOCUMENT_ID,
    FILE_NAME,
    PAGES,
    create_document_chunks
)


def calculate_document_hash(pages):
    """Calculate SHA-256 hash of the document content."""

    document_content = "\n".join(
        f"{page_number}:{page_text}"
        for page_number, page_text in sorted(pages.items())
    )

    return hashlib.sha256(
        document_content.encode("utf-8")
    ).hexdigest()


def persist_document():

    password = getpass.getpass("Enter PostgreSQL password: ")

    document_hash = calculate_document_hash(PAGES)

    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname="supply_chain_ai",
        user="postgres",
        password=password
    ) as connection:

        # Register PostgreSQL vector type with Psycopg.
        register_vector(connection)

        # Check whether this document is already stored.
        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT file_hash
                FROM documents
                WHERE document_id = %s
                """,
                (DOCUMENT_ID,)
            )

            existing_document = cursor.fetchone()

        # Avoid unnecessary embedding generation.
        if (
            existing_document is not None
            and existing_document[0] == document_hash
        ):
            print("Document is already up to date.")
            print("Skipping embedding generation.")
            return

        print("Generating document chunks and embeddings...")

        document_chunks = create_document_chunks(
            DOCUMENT_ID,
            FILE_NAME,
            PAGES
        )

        print("Total chunks generated:", len(document_chunks))

        current_time = datetime.now(timezone.utc)

        # Insert or update the parent document.
        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO documents (
                    document_id,
                    file_name,
                    file_hash,
                    created_at,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, %s)

                ON CONFLICT (document_id)
                DO UPDATE SET
                    file_name = EXCLUDED.file_name,
                    file_hash = EXCLUDED.file_hash,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    DOCUMENT_ID,
                    FILE_NAME,
                    document_hash,
                    current_time,
                    current_time
                )
            )

            # Remove old chunks before inserting updated chunks.
            cursor.execute(
                """
                DELETE FROM document_chunks
                WHERE document_id = %s
                """,
                (DOCUMENT_ID,)
            )

            # Insert the new chunks and their embeddings.
            cursor.executemany(
                """
                INSERT INTO document_chunks (
                    chunk_id,
                    document_id,
                    page_number,
                    chunk_index,
                    content,
                    embedding,
                    created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        record["chunk_id"],
                        record["document_id"],
                        record["page_number"],
                        record["chunk_index"],
                        record["content"],
                        record["embedding"],
                        record["created_at"]
                    )
                    for record in document_chunks
                ]
            )

        # The connection context commits if everything succeeds.
        print("Document and chunks persisted successfully!")


if __name__ == "__main__":
    try:
        persist_document()

    except Exception as error:
        print("Persistence failed:", error)
        raise
