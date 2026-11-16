
import requests

from retrieval_service import retrieve_chunks
from context_assembler import assemble_context


OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
GENERATION_MODEL = "llama3.2:1b"


def add_source_labels(context, sources):
    """
    Add stable source labels to the assembled context.
    These labels will be used for answer citations.
    """

    for index, source in enumerate(sources, start=1):

        source_header = (
            f"[Source: {source['file_name']} | "
            f"Document: {source['document_id']} | "
            f"Page: {source['page_number']} | "
            f"Chunk: {source['chunk_id']}]"
        )

        labelled_header = f"[S{index}] {source_header}"

        context = context.replace(
            source_header,
            labelled_header,
            1
        )

    return context


def generate_answer(question, context):
    """
    Generate an answer using only the retrieved context.
    """

    prompt = f"""
You are a helpful Supply Chain Knowledge Assistant.

Answer the user's question using ONLY the information
provided in the retrieved document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts or missing information.
3. If the answer is not available in the context,
   clearly say that the documents do not contain enough information.
4. Cite the sources using their exact source labels,
   such as [S1] or [S2].
5. Do not create or invent source labels.

Retrieved Document Context:
{context}

User Question:
{question}

Provide a clear and concise answer with citations.
"""

    payload = {
        "model": GENERATION_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2
        }
    }

    response = requests.post(
        OLLAMA_GENERATE_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    return result["response"].strip()


def answer_question(question):
    """
    Complete RAG pipeline:
    Retrieval -> Context Assembly -> LLM -> Citations
    """

    # Step 1: Retrieve relevant document chunks
    chunks = retrieve_chunks(
        query=question,
        top_k=3,
        similarity_threshold=0.65,
        document_id=None
    )

    if not chunks:
        return {
            "question": question,
            "answer": "I could not find relevant information in the available documents.",
            "sources": []
        }

    # Step 2: Assemble retrieved context
    context, sources = assemble_context(
        chunks,
        max_chars=10000
    )

    if not context:
        return {
            "question": question,
            "answer": "I could not prepare relevant document context.",
            "sources": []
        }

    # Step 3: Add source labels
    labelled_context = add_source_labels(
        context,
        sources
    )

    # Step 4: Generate grounded answer
    answer = generate_answer(
        question,
        labelled_context
    )

    # Step 5: Return answer and authoritative source metadata
    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":

    question = "What documents are needed for international shipping?"
    # question = "What is the employee leave policy?"

    result = answer_question(question)

    print("\nUSER QUESTION:")
    print(result["question"])

    print("\nGENERATED ANSWER:")
    print(result["answer"])

    print("\nSOURCE REFERENCES:")

    for index, source in enumerate(result["sources"], start=1):
        print(f"\n[S{index}]")
        print("Document:", source["file_name"])
        print("Document ID:", source["document_id"])
        print("Page:", source["page_number"])
        print("Chunk ID:", source["chunk_id"])
        print("Similarity:", round(source["similarity_score"], 4))
