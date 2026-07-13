"""Vector store backends and the factory that picks one from configuration."""

from .. import config
from .base import DIM_META_KEY, EMBEDDER_META_KEY, VectorStore


def get_store(backend: str | None = None) -> VectorStore:
    """Build the configured vector store. ``backend`` overrides the environment."""
    choice = (backend or config.store_backend()).strip().lower()
    if choice == "sqlite":
        from .sqlite_store import SQLiteVectorStore

        return SQLiteVectorStore(config.sqlite_path())
    if choice == "pgvector":
        url = config.database_url()
        if not url:
            raise RuntimeError("DOCQA_STORE=pgvector requires DATABASE_URL to be set")
        from .pgvector_store import PgVectorStore

        return PgVectorStore(url)
    raise ValueError(f"Unknown store backend {choice!r} (expected 'sqlite' or 'pgvector')")


__all__ = ["DIM_META_KEY", "EMBEDDER_META_KEY", "VectorStore", "get_store"]
