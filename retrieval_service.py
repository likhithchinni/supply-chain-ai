
import getpass
import re
import psycopg

from pgvector.psycopg import register_vector
from document_metadata import generate_embedding


DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "supply_chain_ai",
    "user": "postgres"
}

STOP_WORDS = {
    "what", "which", "when", "where", "who",
    "should", "could", "would", "does", "do",
    "the", "and", "for", "with", "from",
    "are", "is", "was", "were", "how",
    "can", "all", "any", "into", "that",
    "this", "these", "those", "their"
}


QUERY_EXPANSIONS = {
    "receiving": {
        "inbound", "delivery", "appointment",
        "carrier", "seal", "purchase", "order",
        "shipment", "package", "count",
        "paperwork", "damage", "wet",
        "cartons", "broken", "goods"
    },
    "inbound": {
        "receiving", "delivery", "appointment",
        "carrier", "seal", "purchase", "order",
        "shipment", "package", "count",
        "paperwork", "damage", "wet",
        "cartons", "broken"
    }
}



def extract_keywords(text):
    """Extract meaningful words from a question or document."""

    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    return {
        word for word in words
        if len(word) >= 3 and word not in STOP_WORDS
    }



def expand_query_keywords(query):
    """
    Expand query keywords using supply-chain terminology.
    """

    keywords = extract_keywords(query)

    expanded_keywords = set(keywords)

    for keyword in keywords:
        expanded_keywords.update(
            QUERY_EXPANSIONS.get(keyword, set())
        )

    return expanded_keywords



def calculate_keyword_score(query_keywords, content):
    """
    Calculate the proportion of query keywords
    found in a document chunk.
    """

    if not query_keywords:
        return 0.0

    content_words = extract_keywords(content)

    matched_keywords = query_keywords.intersection(content_words)

    return len(matched_keywords) / len(query_keywords)


def retrieve_chunks(
    query,
    top_k=3,
    similarity_threshold=0.65,
    document_id=None,
    neighbor_window=1
):
    """Retrieve and rank chunks using hybrid search."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if not 0 <= similarity_threshold <= 1:
        raise ValueError(
            "Similarity threshold must be between 0 and 1"
        )

    if neighbor_window < 0:
        raise ValueError("neighbor_window cannot be negative")

    query_embedding = generate_embedding(query)

    query_keywords = expand_query_keywords(query)

    # Retrieve more candidates before hybrid ranking
    candidate_limit = top_k * 10

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

    if document_id is not None:
        sql += " AND document_id = %s"
        params.append(document_id)

    sql += """
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    params.extend([query_embedding, candidate_limit])

    password = getpass.getpass("Enter PostgreSQL password: ")

    expanded_chunks = {}

    with psycopg.connect(
        **DB_CONFIG,
        password=password
    ) as connection:

        register_vector(connection)

        with connection.cursor() as cursor:

            # 1. Retrieve vector-search candidates
            cursor.execute(sql, params)

            rows = cursor.fetchall()

            candidates = []

            # 2. Calculate hybrid relevance scores
            for row in rows:

                similarity_score = float(row[6])

                keyword_score = calculate_keyword_score(
                    query_keywords,
                    row[5]
                )

                hybrid_score = (
                    0.7 * similarity_score
                    + 0.3 * keyword_score
                )

                candidates.append({
                    "chunk_id": row[0],
                    "document_id": row[1],
                    "file_name": row[2],
                    "page_number": row[3],
                    "chunk_index": row[4],
                    "content": row[5],
                    "similarity_score": similarity_score,
                    "keyword_score": keyword_score,
                    "hybrid_score": hybrid_score,
                    "is_neighbor": False,
                    "matched_chunk_id": row[0]
                })

            # 3. Rank candidates using hybrid score
            candidates.sort(
                key=lambda chunk: chunk["hybrid_score"],
                reverse=True
            )

            matched_chunks = candidates[:top_k]

            # 4. Expand neighboring chunks
            for matched_chunk in matched_chunks:

                row_document_id = matched_chunk["document_id"]
                row_chunk_index = matched_chunk["chunk_index"]

                cursor.execute(
                    """
                    SELECT
                        chunk_id,
                        document_id,
                        file_name,
                        page_number,
                        chunk_index,
                        content
                    FROM document_chunks
                    WHERE document_id = %s
                      AND chunk_index BETWEEN %s AND %s
                    ORDER BY chunk_index
                    """,
                    (
                        row_document_id,
                        max(1, row_chunk_index - neighbor_window),
                        row_chunk_index + neighbor_window
                    )
                )

                neighbor_rows = cursor.fetchall()

                for neighbor in neighbor_rows:

                    chunk_id = neighbor[0]

                    if chunk_id not in expanded_chunks:
                        expanded_chunks[chunk_id] = {
                            "chunk_id": chunk_id,
                            "document_id": neighbor[1],
                            "file_name": neighbor[2],
                            "page_number": neighbor[3],
                            "chunk_index": neighbor[4],
                            "content": neighbor[5],
                            "similarity_score": None,
                            "keyword_score": None,
                            "hybrid_score": None,
                            "is_neighbor": True,
                            "matched_chunk_id": matched_chunk["chunk_id"]
                        }

                # Preserve actual scores for vector matches
                expanded_chunks[
                    matched_chunk["chunk_id"]
                ] = matched_chunk

    return list(expanded_chunks.values())


if __name__ == "__main__":

    question = (
        "What checks should warehouse staff perform "
        "when receiving inbound goods?"
    )

    results = retrieve_chunks(
        query=question,
        top_k=5,
        similarity_threshold=0.40,
        document_id=None,
        neighbor_window=0
    )

    print("\nQuestion:", question)
    print("\nHybrid-ranked results:")

    for index, result in enumerate(results, start=1):

        print(f"\n--- Result {index} ---")
        print("Chunk:", result["chunk_id"])
        print("File:", result["file_name"])
        print("Vector score:", round(result["similarity_score"], 4))
        print("Keyword score:", round(result["keyword_score"], 4))
        print("Hybrid score:", round(result["hybrid_score"], 4))
        print("Content:", result["content"][:250])
