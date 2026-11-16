
import getpass
import psycopg

from pgvector.psycopg import register_vector
from document_metadata import generate_embedding


def search_documents(query, top_k=3):

    password = getpass.getpass("Enter PostgreSQL password: ")

    # Generate embedding for the user's question.
    query_embedding = generate_embedding(query)

    with psycopg.connect(
        host="localhost",
        port=5432,
        dbname="supply_chain_ai",
        user="postgres",
        password=password
    ) as connection:

        # Register PostgreSQL vector type.
        register_vector(connection)

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    page_number,
                    content,
                    embedding <=> %s::vector AS cosine_distance,
                    1 - (embedding <=> %s::vector) AS similarity_score

                FROM document_chunks

                ORDER BY embedding <=> %s::vector

                LIMIT %s
                """,
                (
                    query_embedding,
                    query_embedding,
                    query_embedding,
                    top_k
                )
            )

            results = cursor.fetchall()

    print("\nSearch query:", query)
    print("\nTop", top_k, "relevant chunks:")

    for rank, result in enumerate(results, start=1):

        (
            chunk_id,
            document_id,
            page_number,
            content,
            distance,
            similarity
        ) = result

        print("\n-----------------------------------")
        print("Rank:", rank)
        print("Chunk ID:", chunk_id)
        print("Document ID:", document_id)
        print("Page Number:", page_number)
        print("Cosine Distance:", round(float(distance), 4))
        print("Similarity Score:", round(float(similarity), 4))
        print("Content:", content)


if __name__ == "__main__":

    query = input("Enter your search question: ")

    search_documents(query, top_k=3)
