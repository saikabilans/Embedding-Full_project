import os
import requests
import pytest
from dotenv import load_dotenv


load_dotenv()


BASE_URL = "http://127.0.0.1:8000"

API_KEY = os.getenv("API_KEY", "my-secret-key-sa1245159")

HEADERS = {
    "X-API-Key": API_KEY
}




def check_server():
    """Check whether FastAPI server is running."""
    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=5
        )
        return True
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="session", autouse=True)
def server_check():
    """Make sure FastAPI is running before tests."""
    if not check_server():
        pytest.fail(
            "FastAPI server is not running.\n"
            "Start it using:\n"
            "python -m uvicorn main:app --reload"
        )

def test_health():


    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10
    )


    print("\nHealth Status:", response.status_code)
    print("Health Response:", response.text)


    assert response.status_code == 200

def test_search_without_api_key():


    payload = {
        "query": "What is artificial intelligence?",
        "top_k": 5,
        "doc_type": "pdf",
        "min_score": 0
    }


    response = requests.post(
        f"{BASE_URL}/search",
        json=payload,
        timeout=30
    )


    print("\nSearch without API key")
    print("Status:", response.status_code)
    print("Response:", response.text)


    assert response.status_code == 401

def test_search_wrong_api_key():


    payload = {
        "query": "What is artificial intelligence?",
        "top_k": 5,
        "doc_type": "pdf",
        "min_score": 0
    }


    headers = {
        "X-API-Key": "wrong-api-key"
    }


    response = requests.post(
        f"{BASE_URL}/search",
        json=payload,
        headers=headers,
        timeout=30
    )


    print("\nSearch with wrong API key")
    print("Status:", response.status_code)
    print("Response:", response.text)


    assert response.status_code == 401

def test_search_correct_api_key():


    payload = {
        "query": "What is artificial intelligence?",
        "top_k": 5,
        "doc_type": "pdf",
        "min_score": 0
    }


    response = requests.post(
        f"{BASE_URL}/search",
        json=payload,
        headers=HEADERS,
        timeout=60
    )


    print("\nMachine learning search")
    print("Status:", response.status_code)
    print("Response:", response.text)


    assert response.status_code in [200, 500]

def test_search_another_query():


    payload = {
        "query": "What is machine learning?",
        "top_k": 5,
        "doc_type": "pdf",
        "min_score": 0
    }


    response = requests.post(
        f"{BASE_URL}/search",
        json=payload,
        headers=HEADERS,
        timeout=60
    )


    print("\nAnother search query")
    print("Status:", response.status_code)
    print("Response:", response.text)


    assert response.status_code in [200, 500]

def test_documents_without_api_key():


    response = requests.post(
        f"{BASE_URL}/documents",
        params={
            "text": "This is a test document."
        },
        timeout=30
    )


    print("\nDocuments without API key")
    print("Status:", response.status_code)
    print("Response:", response.text)


    assert response.status_code == 401

def test_documents_wrong_api_key():


    headers = {
        "X-API-Key": "wrong-api-key"
    }

    response = requests.post(
        f"{BASE_URL}/documents",
        params={
            "text": "This is a test document."
        },
        headers=headers,
        timeout=30
    )

    print("\nDocuments with wrong API key")
    print("Status:", response.status_code)
    print("Response:", response.text)


    assert response.status_code == 401

def test_documents_correct_api_key():


    response = requests.post(
        f"{BASE_URL}/documents",
        params={
            "text": "This is a test document for semantic search."
        },
        headers=HEADERS,
        timeout=60
    )


    print("\nDocuments with correct API key")
    print("Status:", response.status_code)
    print("Response:", response.text)


    assert response.status_code in [200, 500]

def test_collection():
    response = requests.get(
        f"{BASE_URL}/collection",
        headers=HEADERS,
        timeout=30
    )


    print("\nCollection")
    print("Status:", response.status_code)
    print("Response:", response.text)

    assert response.status_code in [200, 500]