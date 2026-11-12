from pathlib import Path
import os
import time
import json
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sklearn.cluster import KMeans
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output" / "task1"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()


sentences = [
    "Python is used for machine learning.",
    "Machine learning helps computers learn from data.",
    "Artificial intelligence can automate many tasks.",
    "Deep learning uses neural networks.",
    "Natural language processing works with human language.",
    "The company provides employee health insurance.",
    "Employees receive annual leave according to company policy.",
    "The office working hours are from 10 AM to 6 PM.",
    "Customers can search products using semantic search.",
    "Vector databases store embeddings for similarity search."
]

data_file = DATA_DIR / "task1_data.csv"

pd.DataFrame({
    "id": range(1, len(sentences) + 1),
    "sentence": sentences
}).to_csv(data_file, index=False)


def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


tests = [
    ([1, 2, 3], [2, 4, 6]),
    ([1, 0], [0, 1]),
    ([1, 0], [-1, 0])
]

for a, b in tests:
    result = cosine_similarity(a, b)
    sklearn_result = float(
        sklearn_cosine([a], [b])[0][0]
    )

    print(
        f"{a} vs {b} -> "
        f"{result:.1f} | sklearn -> {sklearn_result:.1f}"
    )

local_model_name = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

start = time.perf_counter()

local_model = SentenceTransformer(local_model_name)

local_embeddings = local_model.encode(
    sentences,
    convert_to_numpy=True,
    show_progress_bar=False
)

local_time = time.perf_counter() - start

print("Model:", local_model_name)
print("Dimensions:", local_embeddings.shape[1])
print("Time:", round(local_time, 4), "seconds")

np.save(
    MODELS_DIR / "task1_local_embeddings.npy",
    local_embeddings
)

api_embeddings = None
api_time = None
api_error = None

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
)

api_model = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL",
    "openai/text-embedding-3-small"
)

if api_key:
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        start = time.perf_counter()

        response = client.embeddings.create(
            model=api_model,
            input=sentences
        )

        api_time = time.perf_counter() - start

        api_embeddings = np.array(
            [item.embedding for item in response.data]
        )

        print("Model:", api_model)
        print("Dimensions:", api_embeddings.shape[1])
        print("Time:", round(api_time, 4), "seconds")

        np.save(
            MODELS_DIR / "task1_api_embeddings.npy",
            api_embeddings
        )

    except Exception as e:
        api_error = str(e)
        print("OpenRouter API error:", api_error)

else:
    api_error = "OPENROUTER_API_KEY not found in .env"
    print(api_error)


similarity_matrix = sklearn_cosine(local_embeddings)

matrix_file = OUTPUT_DIR / "similarity_matrix.csv"

pd.DataFrame(
    similarity_matrix,
    index=range(1, len(sentences) + 1),
    columns=range(1, len(sentences) + 1)
).to_csv(matrix_file)

print("\nSimilarity matrix saved:")
print(matrix_file)


pairs = []

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = similarity_matrix[i][j]

        pairs.append({
            "sentence_1": sentences[i],
            "sentence_2": sentences[j],
            "score": float(score)
        })

pairs = sorted(
    pairs,
    key=lambda x: x["score"],
    reverse=True
)

top_3 = pairs[:3]

for item in top_3:
    print(
        round(item["score"], 4),
        "|",
        item["sentence_1"],
        "|",
        item["sentence_2"]
    )

queries = [
    "How does machine learning work?",
    "What are employee working hours?",
    "How does vector search work?"
]

for query in queries:

    query_embedding = local_model.encode(
        [query],
        convert_to_numpy=True
    )

    scores = sklearn_cosine(
        query_embedding,
        local_embeddings
    )[0]

    best_index = int(np.argmax(scores))

    print("\nQuery:", query)
    print("Result:", sentences[best_index])
    print("Score:", round(float(scores[best_index]), 4))

duplicate_threshold = 0.85

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):

        score = similarity_matrix[i][j]

        if score >= duplicate_threshold:
            print(
                "Possible duplicate:",
                sentences[i],
                "<->",
                sentences[j],
                "| Score:",
                round(float(score), 4)
            )


kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(local_embeddings)

cluster_names = {
    0: "Technology",
    1: "Company Policy",
    2: "Search and AI"
}

cluster_data = []

for sentence, label in zip(sentences, labels):

    cluster_data.append({
        "sentence": sentence,
        "cluster": int(label),
        "cluster_name": cluster_names.get(
            int(label),
            f"Cluster {label}"
        )
    })

cluster_file = OUTPUT_DIR / "clusters.csv"

pd.DataFrame(cluster_data).to_csv(
    cluster_file,
    index=False
)

for item in cluster_data:
    print(
        item["cluster_name"],
        "->",
        item["sentence"]
    )


report = []

report.append("WEEK 6 TASK 1 - EMBEDDINGS AND COSINE SIMILARITY")
report.append("=" * 55)
report.append("")
report.append("Local model:")
report.append(local_model_name)
report.append(f"Local dimensions: {local_embeddings.shape[1]}")
report.append(f"Local execution time: {local_time:.4f} seconds")
report.append("")

if api_embeddings is not None:
    report.append("OpenRouter model:")
    report.append(api_model)
    report.append(
        f"API dimensions: {api_embeddings.shape[1]}"
    )
    report.append(
        f"API execution time: {api_time:.4f} seconds"
    )
else:
    report.append("OpenRouter API:")
    report.append("Not executed")
    report.append(f"Reason: {api_error}")

report.append("")
report.append("Cosine similarity tests:")
report.append("[1,2,3] vs [2,4,6] -> 1.0")
report.append("[1,0] vs [0,1] -> 0.0")
report.append("[1,0] vs [-1,0] -> -1.0")
report.append("")

report.append("Top 3 similarity pairs:")

for item in top_3:
    report.append(
        f"{item['score']:.4f} | "
        f"{item['sentence_1']} | "
        f"{item['sentence_2']}"
    )

report.append("")
report.append("Local vs API decision:")
report.append(
    "Local models are useful when privacy, offline execution "
    "and predictable local processing are important."
)
report.append(
    "API models are useful when easy access to hosted models "
    "and managed infrastructure are preferred."
)
report.append(
    "Local models avoid API usage costs but require local "
    "CPU/GPU resources."
)
report.append(
    "API models can provide convenient scaling but require "
    "network access and API credentials."
)

report_file = REPORTS_DIR / "task1_report.txt"

report_file.write_text(
    "\n".join(report),
    encoding="utf-8"
)
