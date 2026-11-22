import json
import time

from rag_pipeline import answer_question


QUESTION = (
    "What checks should warehouse staff perform "
    "when receiving inbound goods?"
)

print("Running complete Supply Chain RAG pipeline...")
start = time.perf_counter()

result = answer_question(
    question=QUESTION,
    top_k=2,
    similarity_threshold=0.40,
    neighbor_window=2
)

duration = time.perf_counter() - start

print("\nPIPELINE RESPONSE:")
print(json.dumps(result, indent=4, ensure_ascii=False))
print(f"\nTotal pipeline time: {duration:.2f} seconds")

if result.get("status") != "SUCCESS":
    raise SystemExit("RAG pipeline test failed: " + str(result.get("message")))

if not result.get("answer"):
    raise SystemExit("RAG pipeline returned an empty answer.")

if result.get("citation_status") not in ("VALID",):
    raise SystemExit(
        "RAG pipeline returned an unexpected citation status: "
        + str(result.get("citation_status"))
    )

print("\nRAG PIPELINE TEST PASSED")
