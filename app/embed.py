import os

from sentence_transformers import SentenceTransformer


MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks, model):
    # 1. Create a list containing the content of every chunk
    chunks_content = []
    for chunk in chunks:
        chunks_content.append(chunk['content'])
    

    # 2. Pass that list into model.encode_document()
    #    Use convert_to_numpy=True
    embeddings = model.encode_document(chunks_content,convert_to_numpy=True)


    # 3. Pair each chunk with its corresponding embedding
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
        
    
    return chunks

