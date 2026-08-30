"""Offline ingestion workflow for the enterprise knowledge base.

Run with: python -m app.ingest
Use --force after changing source documents, chunking, or the embedding model.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from app.cache import get_knowledge_fingerprint, load_chunk_cache, save_chunk_cache
from app.chunk import paragraph_chunk_knowledge_base
from app.config import Settings, settings
from app.embed import embed_chunks, load_embedding_model
from app.load import load_knowledge_base
from app.vector_store import get_vector_store


@dataclass
class IngestionResult:
    """Artifacts and facts produced by one ingestion run."""

    chunks: list[dict]
    vector_client: object
    source_document_count: int
    cache_rebuilt: bool


def count_source_documents(knowledge_path: Path) -> int:
    return sum(1 for _ in knowledge_path.rglob("*.md"))


def prepare_index(
    embedding_model,
    runtime_settings: Settings = settings,
    force_rebuild: bool = False,
) -> IngestionResult:
    """Load a valid index or rebuild embeddings and Qdrant when required."""
    fingerprint = get_knowledge_fingerprint(runtime_settings.knowledge_path)
    chunks = None if force_rebuild else load_chunk_cache(
        runtime_settings.cache_path,
        fingerprint,
    )
    source_document_count = count_source_documents(runtime_settings.knowledge_path)
    cache_rebuilt = chunks is None

    if cache_rebuilt:
        documents = load_knowledge_base(runtime_settings.knowledge_path)
        chunks = paragraph_chunk_knowledge_base(documents, chunk_size=100)
        chunks = embed_chunks(chunks, embedding_model)
        save_chunk_cache(chunks, runtime_settings.cache_path, fingerprint)

    vector_client = get_vector_store(
        chunks,
        path=runtime_settings.qdrant_path,
        rebuild=cache_rebuilt,
    )

    return IngestionResult(
        chunks=chunks,
        vector_client=vector_client,
        source_document_count=source_document_count,
        cache_rebuilt=cache_rebuilt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the Enterprise RAG Assistant search index."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild embeddings and the Qdrant collection even if the cache is valid.",
    )
    args = parser.parse_args()

    print("Loading embedding model...")
    embedding_model = load_embedding_model()
    result = prepare_index(embedding_model, force_rebuild=args.force)

    action = "rebuilt" if result.cache_rebuilt else "reused"
    print("\nINGESTION COMPLETE")
    print(f"Source documents: {result.source_document_count}")
    print(f"Indexed chunks: {len(result.chunks)}")
    print(f"Embedding cache: {action}")
    print(f"Qdrant path: {settings.qdrant_path}")
    result.vector_client.close()


if __name__ == "__main__":
    main()
