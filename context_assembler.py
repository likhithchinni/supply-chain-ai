
import re

from retrieval_service import retrieve_chunks


def remove_duplicate_chunks(chunks):
    """Remove duplicate chunk IDs while preserving order."""

    unique_chunks = []
    seen_chunk_ids = set()

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]

        if chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            unique_chunks.append(chunk)

    return unique_chunks


def group_and_sort_chunks(chunks):
    """Group chunks by document and sort by chunk index."""

    grouped = {}

    for chunk in chunks:
        document_id = chunk["document_id"]
        grouped.setdefault(document_id, []).append(chunk)

    ordered_chunks = []

    for document_chunks in grouped.values():
        document_chunks.sort(
            key=lambda chunk: chunk["chunk_index"]
        )
        ordered_chunks.extend(document_chunks)

    return ordered_chunks


def remove_text_overlap(previous_content, current_content):
    """Remove repeated words at consecutive chunk boundaries."""

    previous_words = previous_content.split()
    current_words = current_content.split()

    max_overlap = min(
        len(previous_words),
        len(current_words),
        20
    )

    overlap_length = 0

    for size in range(max_overlap, 0, -1):
        previous_suffix = [
            word.casefold() for word in previous_words[-size:]
        ]

        current_prefix = [
            word.casefold() for word in current_words[:size]
        ]

        if previous_suffix == current_prefix:
            overlap_length = size
            break

    return " ".join(current_words[overlap_length:])


def group_consecutive_chunks(chunks):
    """
    Merge consecutive chunks from the same document and page.

    This prevents sentences from being split across citation labels.
    """

    groups = []

    for chunk in chunks:
        if not groups:
            groups.append([chunk])
            continue

        previous = groups[-1][-1]

        same_document = (
            chunk["document_id"] == previous["document_id"]
        )

        same_page = (
            chunk["page_number"] == previous["page_number"]
        )

        consecutive = (
            chunk["chunk_index"] == previous["chunk_index"] + 1
        )

        if same_document and same_page and consecutive:
            groups[-1].append(chunk)
        else:
            groups.append([chunk])

    return groups


def assemble_context(chunks, max_chars=10000):
    """
    Assemble context using grouped consecutive chunks.

    Each source label represents a complete group of consecutive
    chunks from the same document and page.
    """

    if not chunks:
        return "", []

    # Step 1: Remove duplicate chunks.
    unique_chunks = remove_duplicate_chunks(chunks)

    # Step 2: Sort chunks.
    ordered_chunks = group_and_sort_chunks(unique_chunks)

    # Step 3: Merge consecutive chunks.
    chunk_groups = group_consecutive_chunks(ordered_chunks)

    context_parts = []
    sources = []
    current_length = 0

    # Step 4: Build complete source sections.
    for group in chunk_groups:

        combined_contents = []
        previous_content = None

        for chunk in group:
            original_content = chunk["content"]

            if previous_content is None:
                content = original_content
            else:
                content = remove_text_overlap(
                    previous_content,
                    original_content
                )

            if content.strip():
                combined_contents.append(content.strip())

            previous_content = original_content

        combined_content = " ".join(combined_contents).strip()

        # A chunk can end exactly at the next section heading (for example, "3.").
        # Remove only that dangling heading; keep all complete source sentences.
        combined_content = re.sub(r"\s+\d+[.)]\s*$", "", combined_content).strip()

        if not combined_content:
            continue

        first_chunk = group[0]

        chunk_ids = [
            chunk["chunk_id"] for chunk in group
        ]

        source = {
            "document_id": first_chunk["document_id"],
            "file_name": first_chunk["file_name"],
            "page_number": first_chunk["page_number"],
            "chunk_id": chunk_ids[0],
            "chunk_ids": chunk_ids,
            "similarity_score": max(
                (
                    chunk.get("similarity_score")
                    for chunk in group
                    if chunk.get("similarity_score") is not None
                ),
                default=None
            ),
            "content": combined_content
        }

        source_header = (
            f"[Source: {source['file_name']} | "
            f"Document: {source['document_id']} | "
            f"Page: {source['page_number']}]"
        )

        section = f"{source_header}\n{combined_content}"

        separator_length = 2 if context_parts else 0

        if (
            current_length
            + separator_length
            + len(section)
            > max_chars
        ):
            break

        context_parts.append(section)

        current_length += separator_length + len(section)

        sources.append(source)

    context = "\n\n".join(context_parts)

    return context, sources


if __name__ == "__main__":

    question = (
        "What checks should warehouse staff perform "
        "when receiving inbound goods?"
    )

    retrieved_chunks = retrieve_chunks(
        query=question,
        top_k=2,
        similarity_threshold=0.40,
        document_id=None,
        neighbor_window=2
    )

    context, sources = assemble_context(
        retrieved_chunks,
        max_chars=10000
    )

    print("\nQUESTION:")
    print(question)

    print("\nRETRIEVED CHUNKS:", len(retrieved_chunks))
    print("ASSEMBLED SOURCES:", len(sources))
    print("CONTEXT CHARACTERS:", len(context))

    print("\nASSEMBLED CONTEXT:")
    print(context)

    print("\nSOURCE REFERENCES:")

    for index, source in enumerate(sources, start=1):
        print(f"[S{index}]", source)
