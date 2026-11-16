
from retrieval_service import retrieve_chunks


def remove_duplicate_chunks(chunks):
    """
    Remove duplicate chunks using their unique chunk_id.
    Preserve the original retrieval order.
    """
    unique_chunks = []
    seen_chunk_ids = set()

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]

        if chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            unique_chunks.append(chunk)

    return unique_chunks


def assemble_context(chunks, max_chars=10000):
    """
    Combine retrieved chunks into one context string.

    Preserve source metadata for future citations.
    max_chars limits the context length by characters,
    not by model tokens.
    """

    if not chunks:
        return "", []

    # Step 1: Remove duplicates
    unique_chunks = remove_duplicate_chunks(chunks)

    context_parts = []
    sources = []
    current_length = 0

    # Step 2: Organize chunks into context
    for chunk in unique_chunks:

        source = (
            f"[Source: {chunk['file_name']} | "
            f"Document: {chunk['document_id']} | "
            f"Page: {chunk['page_number']} | "
            f"Chunk: {chunk['chunk_id']}]"
        )

        content = chunk["content"]

        section = f"{source}\n{content}"

        # Step 3: Enforce character limit
        separator_length = 2 if context_parts else 0

        if current_length + separator_length + len(section) > max_chars:
            break

        context_parts.append(section)
        current_length += separator_length + len(section)

        # Step 4: Preserve citation metadata
        sources.append({
            "document_id": chunk["document_id"],
            "file_name": chunk["file_name"],
            "page_number": chunk["page_number"],
            "chunk_id": chunk["chunk_id"],
            "similarity_score": chunk["similarity_score"]
        })

    # Step 5: Combine everything
    context = "\n\n".join(context_parts)

    return context, sources


if __name__ == "__main__":

    question = "What documents are needed for international shipping?"

    # Step 1: Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(
        query=question,
        top_k=3,
        similarity_threshold=0.65,
        document_id=None
    )

    # Step 2: Assemble context
    context, sources = assemble_context(
        retrieved_chunks,
        max_chars=10000
    )

    print("\nUSER QUESTION:")
    print(question)

    print("\nASSEMBLED CONTEXT:")
    print(context)

    print("\nSOURCE REFERENCES:")

    for source in sources:
        print(source)

    print("\nTOTAL SOURCES:", len(sources))
    print("TOTAL CONTEXT CHARACTERS:", len(context))
