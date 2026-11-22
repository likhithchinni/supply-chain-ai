from typing import Optional, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from document_ingestion import process_pdf
from document_repository import persist_document
from rag_pipeline import answer_question


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app = FastAPI(
    title="Supply Chain AI Knowledge Assistant",
    description="REST API for the Supply Chain RAG pipeline",
    version="1.0.0"
)


class RagAskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question to ask the RAG system"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of relevant chunks"
    )
    similarity_threshold: float = Field(
        default=0.40,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Optional document ID filter"
    )


@app.get("/api/rag/health")
def health_check():
    return {
        "status": "UP",
        "service": "Supply Chain RAG API",
        "version": "1.0.0"
    }


@app.post("/api/rag/ask")
def ask_question(request: RagAskRequest) -> dict[str, Any]:
    return answer_question(
        question=request.question,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        document_id=request.document_id
    )


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File size cannot exceed 10 MB")

        if not content.startswith(b"%PDF-"):
            raise HTTPException(status_code=400, detail="Invalid PDF file")

        ingestion_result = process_pdf(content)
        saved_document = persist_document(
            file_name=file.filename,
            pdf_bytes=content,
            chunks=ingestion_result["chunks"]
        )

        return {
            "status": "SUCCESS",
            "message": "PDF processed and stored successfully",
            "document_id": saved_document["document_id"],
            "file_name": saved_document["file_name"],
            "file_size_bytes": len(content),
            "total_pages": ingestion_result["total_pages"],
            "total_chunks": saved_document["total_chunks"]
        }

    except HTTPException:
        raise
    except Exception as error:
        print(f"Document ingestion failed: {error}")
        raise HTTPException(
            status_code=500,
            detail="Document processing or database persistence failed"
        )
    finally:
        await file.close()
