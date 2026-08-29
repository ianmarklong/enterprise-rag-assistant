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

Ollama must be running locally with the Qwen model available:

```powershell
ollama pull qwen3:4b
uvicorn app.api:app --reload
```

Open <http://localhost:8000> in a browser.

## Run with Docker

Start Ollama on the host and ensure its model has been downloaded first:

```powershell
ollama pull qwen3:4b
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

Useful Docker commands:

```powershell
docker compose logs -f
docker compose down
docker compose down --volumes
```

The last command deliberately removes persistent index and model-cache volumes,
so use it only when you intend to rebuild from scratch.
