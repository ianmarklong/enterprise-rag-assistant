import logging
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.cache import get_knowledge_fingerprint, load_chunk_cache, save_chunk_cache
from app.chunk import paragraph_chunk_knowledge_base
from app.config import settings
from app.embed import embed_chunks, load_embedding_model
from app.generate import generate_answer
from app.load import load_knowledge_base
from app.prompt import build_prompt
from app.rerank import load_reranker, rerank
from app.vector_store import get_vector_store, search_vector_store


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise RAG Assistant")
app.mount("/static", StaticFiles(directory=settings.static_path), name="static")


class QueryRequest(BaseModel):
    question: str = Field(max_length=1000)
    category: str | None = Field(default=None, max_length=100)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        return value or None


print("Loading RAG system...")
embedding_model = load_embedding_model()
fingerprint = get_knowledge_fingerprint(settings.knowledge_path)
chunks = load_chunk_cache(settings.cache_path, fingerprint)
cache_rebuilt = False

if chunks is None:
    print("No valid embedding cache found. Rebuilding...")
    documents = load_knowledge_base(settings.knowledge_path)
    chunks = paragraph_chunk_knowledge_base(documents, chunk_size=100)
    chunks = embed_chunks(chunks, embedding_model)
    save_chunk_cache(chunks, settings.cache_path, fingerprint)
    cache_rebuilt = True
    print("Embedding cache saved.")
else:
    print("Loaded embeddings from cache.")

vector_client = get_vector_store(
    chunks,
    path=settings.qdrant_path,
    rebuild=cache_rebuilt
)
reranker = load_reranker()
print("RAG system ready.")


@app.get("/")
def home():
    return FileResponse(settings.static_path / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "retrieval_top_k": settings.retrieval_top_k,
        "rerank_top_k": settings.rerank_top_k
    }


@app.get("/metadata/categories")
def categories():
    return {"categories": sorted({chunk["category"] for chunk in chunks})}


@app.post("/query")
def query(request: QueryRequest):
    question = request.question
    category = request.category
    request_id = uuid4().hex
    started_at = time.perf_counter()

    embedding_started_at = time.perf_counter()
    query_embedding = embedding_model.encode_query(
        question,
        convert_to_numpy=True
    )
    embedding_ms = (time.perf_counter() - embedding_started_at) * 1000

    retrieval_started_at = time.perf_counter()
    candidates = search_vector_store(
        vector_client,
        query_embedding,
        top_k=settings.retrieval_top_k,
        category=category
    )
    retrieval_ms = (time.perf_counter() - retrieval_started_at) * 1000

    if not candidates:
        response = {
            "request_id": request_id,
            "question": question,
            "category": category,
            "answer": "INSUFFICIENT_DOCUMENTATION",
            "sources": []
        }
        logger.info(
            "query_completed request_id=%s question_chars=%s category=%s "
            "candidates=0 results=0 embedding_ms=%.1f retrieval_ms=%.1f total_ms=%.1f",
            request_id,
            len(question),
            category or "all",
            embedding_ms,
            retrieval_ms,
            (time.perf_counter() - started_at) * 1000,
        )
        return response

    rerank_started_at = time.perf_counter()
    results = rerank(
        question,
        candidates,
        reranker,
        top_k=settings.rerank_top_k
    )
    rerank_ms = (time.perf_counter() - rerank_started_at) * 1000
    prompt = build_prompt(question, results)

    generation_started_at = time.perf_counter()
    try:
        answer = generate_answer(prompt)
    except Exception:
        logger.exception("generation_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=503,
            detail="The local model could not generate an answer."
        )
    generation_ms = (time.perf_counter() - generation_started_at) * 1000

    sources = [
        {
            "source": result["source"],
            "score": result["score"],
            "rerank_score": result["rerank_score"]
        }
        for result in results
    ]
    response = {
        "request_id": request_id,
        "question": question,
        "category": category,
        "answer": answer,
        "sources": sources
    }

    logger.info(
        "query_completed request_id=%s question_chars=%s category=%s candidates=%s "
        "results=%s embedding_ms=%.1f retrieval_ms=%.1f rerank_ms=%.1f "
        "generation_ms=%.1f total_ms=%.1f",
        request_id,
        len(question),
        category or "all",
        len(candidates),
        len(results),
        embedding_ms,
        retrieval_ms,
        rerank_ms,
        generation_ms,
        (time.perf_counter() - started_at) * 1000,
    )
    return response
