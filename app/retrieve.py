import numpy as np


def cosine_similarity(vector_a, vector_b):
    # calculate dot product
    dot_product = np.dot(vector_a,vector_b)

    # calculate magnitude of vector_a
    magnitude_a = np.linalg.norm(vector_a)

    # calculate magnitude of vector_b
    magnitude_b = np.linalg.norm(vector_b)

    #unlikely but prevents division by 0
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    # return cosine similarity
    return dot_product/(magnitude_a*magnitude_b)

def retrieve(question, chunks, model, top_k=3):
    query_embedding = model.encode_query(
        question,
        convert_to_numpy=True
    )

    results = []

    # For every chunk:
    #     calculate similarity between
    #     query_embedding and chunk["embedding"]

    for chunk in chunks:
        similarity = cosine_similarity(query_embedding,chunk["embedding"])

    #     save the chunk and its similarity score
        results.append({
            "content": chunk["content"],
            "source": chunk["source"],
            "category": chunk["category"],
            "chunk_id": chunk["chunk_id"],
            "score": similarity
        })

    # Sort results by similarity, highest first
    results.sort(key = lambda result:result['score'],reverse=True)

    # Return only top_k results
    return results[:top_k]