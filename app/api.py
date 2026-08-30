import logging
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app.agent import choose_tool
from app.config import settings
from app.embed import load_embedding_model
from app.generate import generate_answer
from app.ingest import prepare_index
from app.prompt import build_prompt
from app.rerank import load_reranker, rerank
from app.vector_store import search_vector_store


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
ingestion_result = prepare_index(embedding_model)
chunks = ingestion_result.chunks
vector_client = ingestion_result.vector_client
logger.info(
    "index_ready source_documents=%s chunks=%s cache_rebuilt=%s",
    ingestion_result.source_document_count,
    len(chunks),
    ingestion_result.cache_rebuilt,
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

    agent_started_at = time.perf_counter()
    decision = choose_tool(question)
    agent_ms = (time.perf_counter() - agent_started_at) * 1000

    if decision.tool == "list_categories":
        available_categories = sorted({chunk["category"] for chunk in chunks})
        response = {
            "request_id": request_id,
            "question": question,
            "category": category,
            "action": decision.tool,
            "agent_selection": decision.selection_mode,
            "answer": "Available knowledge-base categories: "
            + ", ".join(available_categories),
            "sources": [],
        }
        logger.info(
            "agent_completed request_id=%s tool=%s selection=%s question_chars=%s "
            "agent_ms=%.1f total_ms=%.1f",
            request_id,
            decision.tool,
            decision.selection_mode,
            len(question),
            agent_ms,
            (time.perf_counter() - started_at) * 1000,
        )
        return response

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
            "action": decision.tool,
            "agent_selection": decision.selection_mode,
            "answer": "INSUFFICIENT_DOCUMENTATION",
            "sources": []
        }
        logger.info(
            "query_completed request_id=%s tool=%s selection=%s question_chars=%s category=%s "
            "candidates=0 results=0 agent_ms=%.1f embedding_ms=%.1f "
            "retrieval_ms=%.1f total_ms=%.1f",
            request_id,
            decision.tool,
            decision.selection_mode,
            len(question),
            category or "all",
            agent_ms,
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
        "action": decision.tool,
        "agent_selection": decision.selection_mode,
        "answer": answer,
        "sources": sources
    }

    logger.info(
        "query_completed request_id=%s tool=%s selection=%s question_chars=%s category=%s candidates=%s "
        "results=%s agent_ms=%.1f embedding_ms=%.1f retrieval_ms=%.1f rerank_ms=%.1f "
        "generation_ms=%.1f total_ms=%.1f",
        request_id,
        decision.tool,
        decision.selection_mode,
        len(question),
        category or "all",
        len(candidates),
        len(results),
        agent_ms,
        embedding_ms,
        retrieval_ms,
        rerank_ms,
        generation_ms,
        (time.perf_counter() - started_at) * 1000,
    )
    return response
