# Enterprise RAG Assistant

A learning-focused enterprise Retrieval-Augmented Generation project built in Python.

## Current milestone: V0.2 Local RAG application

The current system:

1. Loads fictional enterprise Markdown documents.
2. Splits documents into overlapping word-based chunks.
3. Generates embeddings with Sentence Transformers.
4. Stores embeddings in embedded, persistent Qdrant.
5. Retrieves the top 8 matching chunks for each question.
6. Reranks candidates with a CrossEncoder and keeps the top 3.
7. Builds a grounded prompt and asks a local Qwen model through Ollama.
8. Serves a FastAPI API and browser chat interface with cited sources.

## Current architecture

Browser → FastAPI → query embedding → Qdrant (top 8) → CrossEncoder reranking (top 3) → prompt → Ollama Qwen → answer and sources

## Run locally (Python)

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Ollama must be running locally with the non-thinking Qwen instruct model
available:

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
uvicorn app.api:app --reload
```

Open <http://localhost:8000> in a browser.

## Run with Docker

Start Ollama on the host and ensure its model has been downloaded first:

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
docker compose up --build
```

Then open <http://localhost:8000>. The first start also downloads the embedding
and reranking models, so it can take longer than later starts.

`compose.yaml` creates two named Docker volumes:

- `rag-data` stores `chunk_cache.pkl` and embedded Qdrant data. This preserves
  expensive indexing work when the application container is recreated.
- `model-cache` stores downloaded Hugging Face models. This prevents downloading
  embedding and reranking models on every image rebuild.

The application container talks to host Ollama through
`host.docker.internal:11434`. That address is supplied through `OLLAMA_HOST`,
so another Ollama server can be used by changing the value in `.env`.

## Runtime configuration and logs

Copy `.env.example` to `.env` to set values locally without committing them.
The application supports these settings:

- `OLLAMA_HOST` and `OLLAMA_MODEL` select the local model service and model.
- `RETRIEVAL_TOP_K` controls how many Qdrant candidates are retrieved.
- `RERANK_TOP_K` controls how many candidates remain after reranking.

`RERANK_TOP_K` cannot exceed `RETRIEVAL_TOP_K`; the application rejects that
invalid configuration during startup. Each completed query produces a log entry
with a request ID and timings for embedding, retrieval, reranking, generation,
and the total request. Question text is intentionally not logged.

## Metadata filtering

Each chunk stores its source document and category. The browser can optionally
send a `category` with a question, and Qdrant applies that metadata constraint
while performing semantic retrieval. Available categories are provided by
`GET /metadata/categories`. If a selected category contains no matching chunks,
the API returns `INSUFFICIENT_DOCUMENTATION` without calling the LLM.

## Agent workflow

Before retrieval, Ollama selects one permitted tool: `search_knowledge_base`
for knowledge-seeking questions or `list_categories` for direct requests to
list category labels. The chosen tool name is validated before execution. If
the model does not return exactly one valid tool call, a deterministic router
selects a safe fallback action instead.

Evaluate this behavior with:

```powershell
python -m evaluation.evaluate_agent
```

The evaluation measures tool-selection accuracy and deliberately simulates a
model failure to verify the fallback path.

## Ingestion workflow

RAG has two separate workflows. Ingestion runs when knowledge changes; querying
runs for every user question:

```text
Ingestion: documents → chunks → embeddings → Qdrant index
Querying: question → query embedding → retrieve → rerank → LLM answer
```

Build or verify the index manually with:

```powershell
python -m app.ingest
```

This reuses the embedding cache when it matches the knowledge base. Source
document changes are detected automatically. Force a complete rebuild after
changing chunking logic or the embedding model, or when you deliberately want
to recreate the index:

```powershell
python -m app.ingest --force
```

The FastAPI server uses this same ingestion preparation code during startup,
which keeps the offline command and deployed application consistent.

The retrieval and answer evaluation scripts use an in-memory Qdrant instance.
They do not modify the persistent index used by the application.

When running with Docker, run the same ingestion workflow inside a short-lived
container instead. It writes to the `rag-data` Docker volume that the API
container reads:

```powershell
docker compose --profile ingestion run --build --rm ingest
```

To deliberately rebuild the Docker index, replace the service's default command
with the explicit force command:

```powershell
docker compose --profile ingestion run --build --rm ingest python -m app.ingest --force
```

Useful Docker commands:

```powershell
docker compose logs -f
docker compose down
docker compose down --volumes
```

The last command deliberately removes persistent index and model-cache volumes,
so use it only when you intend to rebuild from scratch.
