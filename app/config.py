"""Environment-driven runtime settings for the RAG application."""

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, default))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


@dataclass(frozen=True)
class Settings:
    knowledge_path: Path
    cache_path: Path
    qdrant_path: Path
    static_path: Path
    retrieval_top_k: int
    rerank_top_k: int

    @classmethod
    def from_environment(cls) -> "Settings":
        data_path = Path(os.getenv("DATA_PATH", PROJECT_ROOT / "data"))
        retrieval_top_k = _positive_int("RETRIEVAL_TOP_K", 8)
        rerank_top_k = _positive_int("RERANK_TOP_K", 3)

        if rerank_top_k > retrieval_top_k:
            raise ValueError(
                "RERANK_TOP_K cannot be greater than RETRIEVAL_TOP_K"
            )

        return cls(
            knowledge_path=Path(
                os.getenv("KNOWLEDGE_PATH", PROJECT_ROOT / "knowledge_base")
            ),
            cache_path=Path(os.getenv("CACHE_PATH", data_path / "chunk_cache.pkl")),
            qdrant_path=Path(os.getenv("QDRANT_PATH", data_path / "qdrant")),
            static_path=PROJECT_ROOT / "app" / "static",
            retrieval_top_k=retrieval_top_k,
            rerank_top_k=rerank_top_k,
        )


settings = Settings.from_environment()
