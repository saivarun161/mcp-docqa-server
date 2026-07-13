"""The storage contract every vector store backend implements.

Both backends persist the same three things: full documents (for
``fetch_document``), embedded chunks (for similarity search), and a small
metadata table that records which embedder built the index.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..models import Chunk, Document, SearchResult

EMBEDDER_META_KEY = "embedder_id"
DIM_META_KEY = "embedding_dim"


class VectorStore(ABC):
    """Persistence + similarity search over embedded document chunks."""

    backend: str

    @abstractmethod
    def ensure_schema(self, dim: int) -> None:
        """Create tables/indexes if missing. Idempotent; ``dim`` fixes vector width."""

    @abstractmethod
    def get_meta(self, key: str) -> str | None: ...

    @abstractmethod
    def set_meta(self, key: str, value: str) -> None: ...

    @abstractmethod
    def doc_content_hash(self, doc_id: str) -> str | None:
        """Content hash recorded at index time, or None if the doc is not indexed."""

    @abstractmethod
    def upsert_document(
        self,
        doc: Document,
        chunks: list[Chunk],
        vectors: np.ndarray,
        content_hash: str,
    ) -> None:
        """Replace a document and all of its chunks atomically."""

    @abstractmethod
    def search(self, vector: np.ndarray, k: int) -> list[SearchResult]:
        """Top-k chunks by cosine similarity to a unit-norm query vector."""

    @abstractmethod
    def get_document(self, doc_id: str) -> Document | None: ...

    @abstractmethod
    def stats(self) -> dict:
        """Corpus counts plus which backend/embedder built the index."""

    @abstractmethod
    def close(self) -> None: ...

    def guard_embedder(self, embedder_id: str) -> None:
        """Fail loudly if ``embedder_id`` doesn't match the one that built this index.

        Vectors from different embedders live in unrelated spaces; comparing
        them doesn't error at the math level, it just returns nonsense. This
        turns that silent failure into an actionable one.
        """
        stored = self.get_meta(EMBEDDER_META_KEY)
        if stored is not None and stored != embedder_id:
            raise RuntimeError(
                f"This index was built with embedder '{stored}' but the current "
                f"configuration selects '{embedder_id}'. Re-index with "
                f"`docqa-ingest index --force`, or set DOCQA_EMBEDDINGS to match."
            )
