import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv
from fastapi import HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


QDRANT_URL = os.getenv(
    "QDRANT_URL",
    os.getenv("QDRANT_URL", "http://localhost:6333")
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "week6_task1"
)

API_KEY = os.getenv("API_KEY", "")

LOCAL_EMBEDDING_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

TOP_K_DEFAULT = 5



print("Loading embedding model...")

embedding_model = SentenceTransformer(
    LOCAL_EMBEDDING_MODEL
)

print("Embedding model loaded successfully.")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_type: str = "string"
    min_score: float = 0.0



def search_qdrant(
    query: str,
    top_k: int = TOP_K_DEFAULT,
    doc_type: str = "string",
    min_score: float = 0.0
) -> List[Dict[str, Any]]:

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    query_vector = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    search_url = (
        f"{QDRANT_URL}/collections/"
        f"{QDRANT_COLLECTION}/points/search"
    )

    payload = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True,
        "with_vectors": False,
        "score_threshold": min_score
    }

    response = requests.post(
        search_url,
        json=payload,
        timeout=30
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Qdrant search failed: {response.text}"
        )

    data = response.json()

    results = []

    for item in data.get("result", []):

        payload_data = item.get("payload", {})

        results.append(
            {
                "id": item.get("id"),
                "score": round(
                    float(item.get("score", 0)),
                    4
                ),
                "payload": payload_data
            }
        )

    return results


def validate_api_key(api_key: str):

    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API_KEY is not configured in .env"
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return True


def perform_search(
    api_key: str,
    query: str,
    top_k: int = 5,
    doc_type: str = "string",
    min_score: float = 0.0
):

    validate_api_key(api_key)

    start_time = time.perf_counter()

    results = search_qdrant(
        query=query,
        top_k=top_k,
        doc_type=doc_type,
        min_score=min_score
    )

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    return {
        "query": query,
        "top_k": top_k,
        "results": results,
        "result_count": len(results),
        "search_time_ms": round(elapsed_ms, 2)
    }


def check_qdrant_health():

    try:

        response = requests.get(
            f"{QDRANT_URL}/collections",
            timeout=5
        )

        if response.status_code == 200:
            return True

        return False

    except Exception:
        return False


def get_collection_info():

    try:

        url = (
            f"{QDRANT_URL}/collections/"
            f"{QDRANT_COLLECTION}"
        )

        response = requests.get(
            url,
            timeout=10
        )

        if response.status_code != 200:
            return {
                "status": "error",
                "message": response.text
            }

        return response.json()

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }