# Enterprise RAG Assistant

A learning-focused enterprise Retrieval-Augmented Generation project built in Python.

## Current milestone: V0.1 Semantic Retrieval

The current system:

1. Loads fictional enterprise Markdown documents.
2. Splits documents into overlapping word-based chunks.
3. Generates embeddings using Sentence Transformers.
4. Embeds a user query.
5. Calculates cosine similarity against every chunk.
6. Returns the top matching chunks.
7. Constructs a grounded prompt for a future LLM.

## Current architecture

Documents → loading → chunking → embeddings → cosine similarity retrieval → prompt construction

## Project status

This version implements retrieval and prompt construction.  
Local LLM generation, evaluation, improved chunking, Qdrant, FastAPI and Docker will be added in later milestones.

## Run locally

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt