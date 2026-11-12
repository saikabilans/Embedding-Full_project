import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from task4_search_api import (
    API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
    LOCAL_EMBEDDING_MODEL,
    check_qdrant_health,
    get_collection_info,
    perform_search
)


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


app = FastAPI(
    title="Week 6 Semantic Search API",
    description="Semantic Search API using FastAPI, Qdrant and embeddings",
    version="1.0.0"
)


class SearchRequest(BaseModel):

    query: str

    top_k: int = 5

    doc_type: str = "string"

    min_score: float = 0.0


@app.get("/")
def root():

    return {
        "message": "Week 6 Semantic Search API",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():

    qdrant_status = check_qdrant_health()

    return {
        "status": "healthy" if qdrant_status else "unhealthy",
        "qdrant": qdrant_status,
        "collection": QDRANT_COLLECTION,
        "embedding_model": LOCAL_EMBEDDING_MODEL
    }


@app.get("/collection")
def collection():

    return get_collection_info()


@app.post("/search")
def search(
    request: SearchRequest,
    x_api_key: Optional[str] = Header(
        default=None,
        alias="x-api-key"
    )
):

    if not x_api_key:

        raise HTTPException(
            status_code=401,
            detail="Missing API key"
        )

    if not API_KEY:

        raise HTTPException(
            status_code=500,
            detail="API_KEY is not configured in .env"
        )

    if x_api_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    if request.top_k < 1:

        raise HTTPException(
            status_code=400,
            detail="top_k must be greater than 0"
        )

    result = perform_search(
        api_key=x_api_key,
        query=request.query,
        top_k=request.top_k,
        doc_type=request.doc_type,
        min_score=request.min_score
    )

    return result


@app.post("/documents")
def upload_document(
    text: str,
    x_api_key: Optional[str] = Header(
        default=None,
        alias="x-api-key"
    )
):

    if not x_api_key:

        raise HTTPException(
            status_code=401,
            detail="Missing API key"
        )

    if x_api_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    if not text.strip():

        raise HTTPException(
            status_code=400,
            detail="Document text cannot be empty"
        )

    return {
        "message": "Document endpoint is working",
        "document_length": len(text),
        "status": "received"
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )