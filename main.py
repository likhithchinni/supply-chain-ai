
from typing import Optional, Any

from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

from rag_pipeline import answer_question


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Supply Chain AI Knowledge Assistant",
    description="REST API for the Supply Chain RAG pipeline",
    version="1.0.0"
)


# --------------------------------------------------
# Request DTO
# --------------------------------------------------

class RagAskRequest(BaseModel):

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to ask the RAG system"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of relevant chunks"
    )

    similarity_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )

    document_id: Optional[str] = Field(
        default=None,
        description="Optional document ID filter"
    )


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/api/rag/health")
def health_check():

    return {
        "status": "UP",
        "service": "Supply Chain RAG API",
        "version": "1.0.0"
    }


# --------------------------------------------------
# Ask endpoint
# --------------------------------------------------

@app.post("/api/rag/ask")
def ask_question(request: RagAskRequest) -> dict[str, Any]:

    result = answer_question(
        question=request.question,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        document_id=request.document_id
    )

    return result
