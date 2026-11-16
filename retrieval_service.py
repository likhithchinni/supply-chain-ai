
import getpass
import psycopg

from pgvector.psycopg import register_vector
from document_metadata import generate_embedding


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "supply_chain_ai",
    "user": "postgres"
}


def retrieve_chunks(
    query,
    top_k=3,
    similarity_threshold=0.65,
    document_id=None
):
    """
    Retrieve relevant document chunks using vector similarity.

    Args:
        query: User's natural-language question.
        top_k: Maximum number of chunks to return.
        similarity_threshold: Minimum cosine similarity (0 to 1).
        document_id: Optional document ID filter.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if not 0 <= similarity_threshold <= 1:
        raise ValueError("Similarity threshold must be between 0 and 1")

    # 1. Convert the question into a 768-dimensional embedding
    query_embedding = generate_embedding(query)

    # 2. Build SQL conditions
    sql = """
        SELECT
            chunk_id,
            document_id,
            file_name,
            page_number,
            chunk_index,
            content,
            1 - (embedding <=> %s::vector) AS similarity_score
        FROM document_chunks
        WHERE 1 - (embedding <=> %s::vector) >= %s
    """

    params = [
        query_embedding,
        query_embedding,
        similarity_threshold
    ]

    # 3. Apply optional document filtering
    if document_id is not None:
        sql += " AND document_id = %s"
        params.append(document_id)

    # 4. Sort by closest vector and limit the results
    sql += """
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    params.extend([query_embedding, top_k])

    # 5. Connect to PostgreSQL and execute the query
    password = getpass.getpass("Enter PostgreSQL password: ")

    with psycopg.connect(
        **DB_CONFIG,
        password=password
    ) as connection:

        register_vector(connection)

        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

    # 6. Convert database rows into reusable Python objects
    results = []

    for row in rows:
        results.append({
            "chunk_id": row[0],
            "document_id": row[1],
            "file_name": row[2],
            "page_number": row[3],
            "chunk_index": row[4],
            "content": row[5],
            "similarity_score": float(row[6])
        })

    return results


if __name__ == "__main__":

    question = "What documents are needed for international shipping?"

    results = retrieve_chunks(
        query=question,
        top_k=3,
        similarity_threshold=0.65,
        document_id=None
    )

    print("\nQuestion:", question)
    print("\nRetrieved chunks:", len(results))

    for index, result in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print("Chunk ID:", result["chunk_id"])
        print("Document ID:", result["document_id"])
        print("File:", result["file_name"])
        print("Page:", result["page_number"])
        print("Similarity:", round(result["similarity_score"], 4))
        print("Content:", result["content"])
