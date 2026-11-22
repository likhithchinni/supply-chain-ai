import re
import requests
import psycopg

from retrieval_service import retrieve_chunks
from context_assembler import assemble_context


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_TIMEOUT_SECONDS = 210

STOP_WORDS = {
    "what", "which", "when", "where", "who", "should", "could", "would",
    "does", "do", "the", "and", "for", "with", "from", "are", "is", "was",
    "were", "how", "can", "all", "any", "into", "that", "this", "these",
    "those", "their", "have", "has", "had", "your", "you", "they", "them",
    "about", "also", "then", "than", "such", "only", "using", "used"
}


def add_source_labels(context, sources):
    """Add stable labels to the context while keeping source metadata separate."""
    labelled_sections = []
    for index, source in enumerate(sources, start=1):
        content = str(source.get("content") or "").strip()
        if content:
            labelled_sections.append(f"[S{index}]\n{content}")
    return "\n\n".join(labelled_sections)


def _tokens(text):
    return {
        token for token in re.findall(r"[a-z0-9]+", (text or "").casefold())
        if len(token) > 2 and token not in STOP_WORDS
    }


def _split_answer_into_items(answer):
    """Normalize numbered, bulleted, or plain-text model output into checklist items."""
    text = re.sub(
        r"^```(?:text|markdown)?\s*|\s*```$",
        "",
        (answer or "").strip(),
        flags=re.IGNORECASE
    )
    text = re.sub(r"\[(?:S\d+)\]", "", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s*$", "", text).strip()

    # Prefer explicit list entries; otherwise treat non-empty paragraphs/lines as items.
    matches = re.findall(
        r"(?ms)^\s*(?:\d+[.)]|[-*])\s+(.+?)(?=^\s*(?:\d+[.)]|[-*])\s+|\Z)",
        text
    )
    if matches:
        raw_items = matches
    else:
        raw_items = [
            part.strip()
            for part in re.split(r"(?:\r?\n)+|(?<=[.!?])\s+(?=[A-Z])", text)
            if part.strip()
        ]

    items = []
    seen = set()
    for raw in raw_items:
        item = re.sub(r"\s+", " ", raw).strip()
        item = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", item).strip()
        item = re.sub(r"[\s.;,:-]+$", "", item).strip()
        if not item:
            continue
        normalized = re.sub(r"[^a-z0-9]+", " ", item.casefold()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            items.append(item)
    return items


def _best_source_labels(item, sources):
    """Choose source labels using content overlap, not model-generated citation syntax."""
    item_tokens = _tokens(item)
    if not item_tokens:
        return ["S1"]

    scored = []
    for index, source in enumerate(sources, start=1):
        source_tokens = _tokens(source.get("content", ""))
        if not source_tokens:
            continue
        overlap = len(item_tokens & source_tokens)
        # Coverage of the answer's meaningful terms is more useful than raw document size.
        score = overlap / len(item_tokens)
        if score > 0:
            scored.append((score, overlap, index))

    if not scored:
        # Retrieval already ranked these sources; use the strongest retrieved source
        # rather than returning an answer with missing citations.
        best_index = max(
            range(1, len(sources) + 1),
            key=lambda i: float(sources[i - 1].get("similarity_score") or 0.0)
        )
        return [f"S{best_index}"]

    scored.sort(reverse=True)
    best_score = scored[0][0]
    # Include close supporting sources, but avoid citing every retrieved document.
    selected = [
        f"S{index}" for score, overlap, index in scored
        if score >= max(0.20, best_score * 0.75) and overlap >= 2
    ]
    return selected[:2] or [f"S{scored[0][2]}"]


def generate_answer(question, context, sources=None):
    """
    Generate an answer and assign citations in Python.
    Citation correctness no longer depends on the small local model following
    citation-format instructions.
    """
    valid_labels = set(re.findall(r"\[(S\d+)\]", context or ""))
    if not valid_labels:
        raise ValueError("No citation labels found in document context.")

    prompt = f"""
You are a Supply Chain Knowledge Assistant.
Answer the user's question using ONLY the document context below.

Rules:
- Include all distinct procedures that directly answer the question.
- Do not invent facts or use outside knowledge.
- Combine related facts and remove repetition.
- Do not include unrelated procedures.
- If the context has no relevant information, respond exactly:
  I could not find relevant information in the available documents.
- Return a concise numbered checklist.
- Do not write source citations; the application adds and validates citations.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 500}
    }

    response = requests.post(
        OLLAMA_GENERATE_URL,
        json=payload,
        timeout=OLLAMA_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    raw_answer = str(response.json().get("response") or "").strip()

    if not raw_answer:
        raise ValueError("Ollama returned an empty response.")

    fallback = "I could not find relevant information in the available documents."
    if fallback.casefold() in raw_answer.casefold():
        return fallback

    items = _split_answer_into_items(raw_answer)
    if not items:
        raise ValueError("Ollama returned text, but no usable answer items could be extracted.")

    # Keep citations deterministic and grounded in the retrieved source content.
    if sources:
        output = []
        for index, item in enumerate(items, start=1):
            labels = _best_source_labels(item, sources)
            citation_text = " ".join(f"[{label}]" for label in labels)
            output.append(f"{index}. {item} {citation_text}")
        return "\n\n".join(output)

    # Compatibility fallback for callers that only pass context.
    only_label = sorted(valid_labels, key=lambda label: int(label[1:]))[0]
    return "\n\n".join(
        f"{index}. {item} [{only_label}]"
        for index, item in enumerate(items, start=1)
    )


def validate_citations(answer, sources):
    found = re.findall(r"\[S(\d+)\]", answer or "")
    valid_source_ids = {str(index) for index in range(1, len(sources) + 1)}
    valid_citations = []
    invalid_citations = []

    for citation_id in found:
        citation = f"[S{citation_id}]"
        target = valid_citations if citation_id in valid_source_ids else invalid_citations
        if citation not in target:
            target.append(citation)

    if not found:
        status = "MISSING_CITATIONS"
    elif invalid_citations:
        status = "INVALID_CITATIONS"
    else:
        status = "VALID"

    return {
        "citation_status": status,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations
    }


def create_error_response(error_code, message, question=None):
    return {
        "status": "ERROR",
        "error_code": error_code,
        "message": message,
        "question": question,
        "answer": None,
        "citation_status": None,
        "citations": [],
        "invalid_citations": []
    }


def answer_question(
    question,
    top_k=5,
    similarity_threshold=0.40,
    document_id=None,
    neighbor_window=2
):
    """Run retrieval, context assembly, local generation, and citation mapping."""
    if not question or not question.strip():
        return create_error_response(
            "INVALID_QUESTION", "Question cannot be empty.", question
        )

    try:
        chunks = retrieve_chunks(
            query=question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            document_id=document_id,
            neighbor_window=neighbor_window
        )
    except requests.exceptions.Timeout:
        return create_error_response(
            "OLLAMA_TIMEOUT", "Ollama embedding request timed out.", question
        )
    except requests.exceptions.ConnectionError:
        return create_error_response(
            "OLLAMA_UNAVAILABLE", "Could not connect to Ollama embedding service.", question
        )
    except psycopg.Error:
        return create_error_response(
            "DATABASE_ERROR", "Could not retrieve documents from PostgreSQL.", question
        )
    except ValueError as error:
        return create_error_response("EMBEDDING_ERROR", str(error), question)
    except Exception as error:
        return create_error_response("RETRIEVAL_ERROR", str(error), question)

    if not chunks:
        return {
            "status": "NO_RELEVANT_DOCUMENTS",
            "error_code": None,
            "message": "No relevant documents found.",
            "question": question,
            "answer": "I could not find relevant information in the available documents.",
            "citation_status": "NOT_APPLICABLE",
            "citations": [],
            "invalid_citations": []
        }

    try:
        context, sources = assemble_context(chunks)
        if not context or not context.strip() or not sources:
            return create_error_response(
                "CONTEXT_ERROR",
                "Could not assemble context or source metadata from retrieved documents.",
                question
            )
        labelled_context = add_source_labels(context, sources)
    except Exception as error:
        return create_error_response("CONTEXT_ERROR", str(error), question)

    try:
        answer = generate_answer(
            question=question,
            context=labelled_context,
            sources=sources
        )
    except requests.exceptions.Timeout:
        return create_error_response(
            "OLLAMA_TIMEOUT", "Ollama generation request timed out.", question
        )
    except requests.exceptions.ConnectionError:
        return create_error_response(
            "OLLAMA_UNAVAILABLE", "Could not connect to Ollama generation service.", question
        )
    except requests.exceptions.RequestException as error:
        return create_error_response("GENERATION_HTTP_ERROR", str(error), question)
    except ValueError as error:
        return create_error_response("ANSWER_GENERATION_ERROR", str(error), question)
    except Exception as error:
        return create_error_response("GENERATION_ERROR", str(error), question)

    citation_result = validate_citations(answer, sources)
    citations = []

    for citation_label in citation_result["valid_citations"]:
        source_index = int(citation_label[2:-1]) - 1
        source = sources[source_index]
        citations.append({
            "citation": citation_label,
            "document_id": source.get("document_id"),
            "file_name": source.get("file_name"),
            "page_number": source.get("page_number"),
            "chunk_id": source.get("chunk_id"),
            "chunk_ids": source.get("chunk_ids", []),
            "similarity_score": source.get("similarity_score")
        })

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
    import json
    test_question = "What documents are needed for international shipping?"
    print(json.dumps(answer_question(test_question), indent=4))
