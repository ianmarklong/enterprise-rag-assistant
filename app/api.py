from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.load import load_knowledge_base
from app.chunk import paragraph_chunk_knowledge_base
from app.embed import load_embedding_model, embed_chunks
from app.vector_store import get_vector_store, search_vector_store
from app.rerank import load_reranker, rerank
from app.prompt import build_prompt
from app.generate import generate_answer
from app.cache import (
    get_knowledge_fingerprint,
    load_chunk_cache,
    save_chunk_cache
)

app = FastAPI(
    title="Enterprise RAG Assistant"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "knowledge_base"
CACHE_PATH = PROJECT_ROOT / "data" / "chunk_cache.pkl"
STATIC_PATH = PROJECT_ROOT / "app" / "static"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_PATH),
    name="static"
)

class QueryRequest(BaseModel):
    question: str

print("Loading RAG system...")

embedding_model = load_embedding_model()

fingerprint = get_knowledge_fingerprint(
    KNOWLEDGE_PATH
)

chunks = load_chunk_cache(
    CACHE_PATH,
    fingerprint
)

cache_rebuilt = False

if chunks is None:
    print("No valid embedding cache found. Rebuilding...")

    documents = load_knowledge_base(
        KNOWLEDGE_PATH
    )

    chunks = paragraph_chunk_knowledge_base(
        documents,
        chunk_size=100
    )

    chunks = embed_chunks(
        chunks,
        embedding_model
    )

    save_chunk_cache(
        chunks,
        CACHE_PATH,
        fingerprint
    )

    cache_rebuilt = True

    print("Embedding cache saved.")

else:
    print("Loaded embeddings from cache.")

vector_client = get_vector_store(chunks,rebuild=cache_rebuilt)

reranker = load_reranker()

print("RAG system ready.")

@app.get("/")
def home():
    return FileResponse(
        STATIC_PATH / "index.html"
    )

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/query")
def query(request: QueryRequest):
    question = request.question

    query_embedding = embedding_model.encode_query(
        question,
        convert_to_numpy=True
    )

    candidates = search_vector_store(
        vector_client,
        query_embedding,
        top_k=8
    )

    results = rerank(
        question,
        candidates,
        reranker,
        top_k=3
    )

    prompt = build_prompt(
        question,
        results
    )

    answer = generate_answer(prompt)

    sources = []

    for result in results:
        sources.append({
            "source": result["source"],
            "score": result["score"],
            "rerank_score": result["rerank_score"]
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }