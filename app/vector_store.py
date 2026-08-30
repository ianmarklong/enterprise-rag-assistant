from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


COLLECTION_NAME = "northstar_chunks"


def create_vector_store(chunks):
    client = QdrantClient(path="data/qdrant") #local embedded Qdrant

    vector_size = len(chunks[0]["embedding"])

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

    points = []

    for point_id, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id = point_id,
                vector = chunk["embedding"].tolist(), #converts the NumPy array into a normal Python list because Qdrant expects the vector in a serializable list-like form
                payload={
                        "content": chunk["content"],
                        "source": chunk["source"],
                        "category": chunk["category"],
                        "chunk_id": chunk["chunk_id"]
                    }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return client

def search_vector_store(client, query_embedding, top_k=3, category=None):
    query_filter = None

    if category is not None:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category)

                )
            ]
        )
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        query_filter=query_filter,
        limit=top_k
    )

    results = []

    for point in response.points:
        results.append({
            "content": point.payload["content"],
            "source": point.payload["source"],
            "category": point.payload["category"],
            "chunk_id": point.payload["chunk_id"],
            "score": point.score
        })

    return results

def get_vector_store(chunks, path, rebuild=False):
    client = QdrantClient(path=str(Path(path)))

    collection_exists = client.collection_exists(
        COLLECTION_NAME
    )

    if rebuild or not collection_exists:
        vector_size = len(chunks[0]["embedding"])

        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        points = []

        for point_id, chunk in enumerate(chunks):
            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk["embedding"].tolist(),
                    payload={
                        "content": chunk["content"],
                        "source": chunk["source"],
                        "category": chunk["category"],
                        "chunk_id": chunk["chunk_id"]
                    }
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        print("Qdrant collection rebuilt.")

    else:
        print("Loaded existing Qdrant collection.")

    return client
