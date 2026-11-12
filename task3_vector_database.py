from pathlib import Path
import json
import time

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance,VectorParams,PointStruct,Filter,FieldCondition,MatchValue,Range)


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output" / "task3"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


dimension = 384
number_of_vectors = 50000

np.random.seed(42)

vectors = np.random.random(
    (number_of_vectors, dimension)
).astype("float32")

faiss.normalize_L2(vectors)

query = np.random.random(
    (1, dimension)
).astype("float32")

faiss.normalize_L2(query)


exact_index = faiss.IndexFlatIP(
    dimension
)

exact_index.add(vectors)

start = time.perf_counter()

exact_scores, exact_ids = exact_index.search(
    query,
    10
)

exact_time = (
    time.perf_counter() - start
) * 1000



hnsw_index = faiss.IndexHNSWFlat(
    dimension,
    32
)

hnsw_index.hnsw.efSearch = 64

hnsw_index.add(vectors)

start = time.perf_counter()

hnsw_scores, hnsw_ids = hnsw_index.search(
    query,
    10
)

hnsw_time = (
    time.perf_counter() - start
) * 1000


exact_set = set(
    exact_ids[0]
)

hnsw_set = set(
    hnsw_ids[0]
)

recall = len(
    exact_set.intersection(hnsw_set)
) / 10


print(
    f"Exact search: {exact_time:.2f} ms"
)

print(
    f"HNSW search: {hnsw_time:.2f} ms"
)

print(
    f"Recall@10: {recall:.2f}"
)


faiss_file = MODELS_DIR / "faiss_hnsw.index"

faiss.write_index(
    hnsw_index,
    str(faiss_file)
)


try:

    client = QdrantClient(
        url="http://localhost:6333"
    )

    collection_name = (
        "week6_assessment"
    )

    existing_collections = [
        c.name
        for c in client.get_collections()
        .collections
    ]

    if collection_name in existing_collections:

        client.delete_collection(
            collection_name
        )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )


    client.update_collection(
        collection_name=collection_name,
        hnsw_config={
            "m": 16,
            "ef_construct": 100
        }
    )

    chunks_file = (
        BASE_DIR /
        "models" /
        "chunks.json"
    )

    chunks = json.loads(
        chunks_file.read_text(
            encoding="utf-8"
        )
    )

    chunks = chunks[:120]

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    embeddings = embeddings.astype(
        "float32"
    )

    points = []

    for i, chunk in enumerate(chunks):

        payload = {
            "source": chunk["source"],
            "page": chunk["page"],
            "section": chunk["section"],
            "doc_type": chunk["doc_type"],
            "text": chunk["text"],
            "chunk_index": chunk["chunk_index"]
        }

        points.append(
            PointStruct(
                id=i + 1,
                vector=embeddings[i].tolist(),
                payload=payload
            )
        )


    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True
    )

    count_run1 = client.count(
        collection_name=collection_name,
        exact=True
    ).count

    client.upsert(
        collection_name=collection_name,
        points=points,
        wait=True
    )

    count_run2 = client.count(
        collection_name=collection_name,
        exact=True
    ).count

    query_vector = embeddings[0].tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=5,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_type",
                    match=MatchValue(
                        value="pdf"
                    )
                ),
                FieldCondition(
                    key="page",
                    range=Range(
                        gte=1
                    )
                )
            ]
        )
    ).points


    source_to_delete = chunks[0]["source"]

    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(
                        value=source_to_delete
                    )
                )
            ]
        ),
        wait=True
    )

    count_after_delete = client.count(
        collection_name=collection_name,
        exact=True
    ).count

    print(
        "Run 1 count:",
        count_run1
    )

    print(
        "Run 2 count:",
        count_run2
    )

    print(
        "Filtered results:",
        len(results)
    )

    print(
        "Count after delete:",
        count_after_delete
    )

    qdrant_status = "SUCCESS"

except Exception as e:

    print(
        "Qdrant error:",
        e
    )

    count_run1 = 0
    count_run2 = 0
    count_after_delete = 0
    qdrant_status = str(e)


report = f"""

FAISS

Vectors:
50000

Dimension:
384

Exact search:
{exact_time:.2f} ms

HNSW search:
{hnsw_time:.2f} ms

Recall@10:
{recall:.2f}

QDRANT

Status:
{qdrant_status}

Run 1 vector count:
{count_run1}

Run 2 vector count:
{count_run2}

Filtered search results:
{len(results) if 'results' in locals() else 0}

Count after deletion:
{count_after_delete}

Re-upsert does not create duplicate points because
the same point IDs are used.

ANN:
Approximate Nearest Neighbor search reduces search
time by avoiding an exhaustive comparison with every
vector.

HNSW:
Hierarchical Navigable Small World is a graph-based
indexing method used for fast approximate similarity
search.

Qdrant:
Useful when application-level vector storage,
metadata filtering and upsert operations are required.
"""

report_file = (
    REPORTS_DIR /
    "task3_report.txt"
)

report_file.write_text(
    report.strip(),
    encoding="utf-8"
)