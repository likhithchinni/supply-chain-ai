
import re
import requests
import psycopg

from retrieval_service import retrieve_chunks
from context_assembler import assemble_context


# Ollama generation API
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

# Generation model
OLLAMA_MODEL = "llama3.2:1b"


def add_source_labels(context, sources):
    """
    Add source labels such as [S1], [S2] to the context.
    """
    labelled_sections = []

    for index, source in enumerate(sources, start=1):
        label = f"[S{index}]"

        section = (
            f"{label}\n"
            f"File: {source.get('file_name')}\n"
            f"Document ID: {source.get('document_id')}\n"
            f"Page: {source.get('page_number')}\n"
            f"Chunk ID: {source.get('chunk_id')}\n"
            f"{source.get('content', '')}"
        )

        labelled_sections.append(section)

    return "\n\n".join(labelled_sections)


def generate_answer(question, context):
    """
    Generate an answer using Ollama.

    Requests exceptions are intentionally allowed to propagate
    so answer_question() can handle them.
    """

    prompt = f"""
You are a Supply Chain Knowledge Assistant.

Answer the question using ONLY the context provided below.

Rules:
1. Do not use external knowledge.
2. Do not invent information.
3. If the answer is unavailable in the context, say:
   "I could not find relevant information in the available documents."
4. Include source labels such as [S1] or [S2] for factual claims.
5. Give a clear and concise answer.

Context:
{context}

Question:
{question}

Answer:
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    response = requests.post(
        OLLAMA_GENERATE_URL,
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    result = response.json()

    answer = result.get("response", "").strip()

    if not answer:
        raise ValueError("LLM returned an empty response")

    return answer


def validate_citations(answer, sources):
    """
    Validate whether source labels in the answer
    refer to sources that were actually retrieved.
    """

    citation_pattern = r"\[S(\d+)\]"

    found_citations = re.findall(citation_pattern, answer)

    valid_source_ids = {
        str(index)
        for index in range(1, len(sources) + 1)
    }

    valid_citations = []
    invalid_citations = []

    for citation_id in found_citations:
        if citation_id in valid_source_ids:
            citation = f"[S{citation_id}]"

            if citation not in valid_citations:
                valid_citations.append(citation)
        else:
            citation = f"[S{citation_id}]"

            if citation not in invalid_citations:
                invalid_citations.append(citation)

    if not found_citations:
        citation_status = "MISSING_CITATIONS"

    elif invalid_citations:
        citation_status = "INVALID_CITATIONS"

    else:
        citation_status = "VALID"

    return {
        "citation_status": citation_status,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations
    }


def create_error_response(error_code, message):
    """
    Return a consistent error response.
    """

    return {
        "status": "ERROR",
        "error_code": error_code,
        "message": message,
        "question": None,
        "answer": None,
        "citation_status": None,
        "citations": [],
        "invalid_citations": []
    }


def answer_question(
    question,
    top_k=3,
    similarity_threshold=0.55,
    document_id=None
):
    """
    Complete RAG pipeline:

    1. Retrieve relevant chunks.
    2. Assemble context.
    3. Generate answer.
    4. Validate citations.
    5. Return structured response.
    """

    # Step 1: Validate question
    if not question or not question.strip():
        return create_error_response(
            "INVALID_QUESTION",
            "Question cannot be empty."
        )

    # Step 2: Retrieve relevant chunks
    try:
        chunks = retrieve_chunks(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            document_id=document_id
        )

    except requests.exceptions.Timeout:
        return create_error_response(
            "OLLAMA_TIMEOUT",
            "Ollama embedding request timed out."
        )

    except requests.exceptions.ConnectionError:
        return create_error_response(
            "OLLAMA_UNAVAILABLE",
            "Could not connect to Ollama embedding service."
        )

    except psycopg.Error:
        return create_error_response(
            "DATABASE_ERROR",
            "Could not retrieve documents from PostgreSQL."
        )

    except ValueError as error:
        return create_error_response(
            "EMBEDDING_ERROR",
            str(error)
        )

    except Exception as error:
        return create_error_response(
            "RETRIEVAL_ERROR",
            str(error)
        )

    # Step 3: Check retrieval result
    if not chunks:
        return {
            "status": "NO_RELEVANT_DOCUMENTS",
            "error_code": None,
            "message": "No relevant documents found.",
            "question": question,
            "answer": (
                "I could not find relevant information "
                "in the available documents."
            ),
            "citation_status": "NOT_APPLICABLE",
            "citations": [],
            "invalid_citations": []
        }

    # Step 4: Assemble context
    try:
        context, sources = assemble_context(chunks)

        if not context or not context.strip():
            return create_error_response(
                "CONTEXT_ERROR",
                "Could not assemble context from retrieved documents."
            )

        if not sources:
            return create_error_response(
                "CONTEXT_ERROR",
                "No source metadata available for the context."
            )

        labelled_context = add_source_labels(context, sources)

    except Exception as error:
        return create_error_response(
            "CONTEXT_ERROR",
            str(error)
        )

    # Step 5: Generate answer
    try:
        answer = generate_answer(
            question=question,
            context=labelled_context
        )

    except requests.exceptions.Timeout:
        return create_error_response(
            "OLLAMA_TIMEOUT",
            "Ollama generation request timed out."
        )

    except requests.exceptions.ConnectionError:
        return create_error_response(
            "OLLAMA_UNAVAILABLE",
            "Could not connect to Ollama generation service."
        )

    except requests.exceptions.RequestException as error:
        return create_error_response(
            "GENERATION_HTTP_ERROR",
            str(error)
        )

    except ValueError as error:
        return create_error_response(
            "EMPTY_LLM_RESPONSE",
            str(error)
        )

    except Exception as error:
        return create_error_response(
            "GENERATION_ERROR",
            str(error)
        )

    # Step 6: Validate citations
    citation_result = validate_citations(
        answer,
        sources
    )

    # Step 7: Map valid citation labels to source metadata
    citations = []

    for citation_label in citation_result["valid_citations"]:
        source_index = int(
            citation_label.replace("[S", "").replace("]", "")
        ) - 1

        source = sources[source_index]

        citations.append({
            "citation": citation_label,
            "document_id": source.get("document_id"),
            "file_name": source.get("file_name"),
            "page_number": source.get("page_number"),
            "chunk_id": source.get("chunk_id"),
            "similarity_score": source.get("similarity_score")
        })

    # Step 8: Return structured response
    return {
        "status": "SUCCESS",
        "error_code": None,
        "message": "Answer generated successfully.",
        "question": question,
        "answer": answer,
        "citation_status": citation_result["citation_status"],
        "citations": citations,
        "invalid_citations": citation_result["invalid_citations"]
    }


if __name__ == "__main__":

    question = "What documents are needed for international shipping?"
    # question = "What is the employee annual leave policy?"
    
    result = answer_question(question)

    import json

    print(json.dumps(result, indent=4))
