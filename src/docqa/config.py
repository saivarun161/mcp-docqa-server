"""Environment-driven configuration.

Every setting is read through a function (not a module constant) so tests and
long-running processes always see the current environment. A local `.env` file
is loaded once at import; real environment variables take precedence.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_SQLITE_PATH = "data/index.db"


def store_backend() -> str:
    """Which vector store to use: ``sqlite`` (default, embedded) or ``pgvector``."""
    return os.getenv("DOCQA_STORE", "sqlite").strip().lower()


def sqlite_path() -> str:
    """Filesystem path of the embedded SQLite index."""
    return os.getenv("DOCQA_SQLITE_PATH", DEFAULT_SQLITE_PATH)


def database_url() -> str | None:
    """Postgres connection string, required when DOCQA_STORE=pgvector."""
    return os.getenv("DATABASE_URL")


def embeddings_backend() -> str:
    """Which embedder to use: ``openai``, ``hash``, or ``auto`` (the default).

    ``auto`` resolves to ``openai`` when an OPENAI_API_KEY is present and to the
    dependency-free ``hash`` fallback otherwise, so a fresh clone works with no
    credentials at all.
    """
    value = os.getenv("DOCQA_EMBEDDINGS", "auto").strip().lower()
    if value == "auto":
        return "openai" if os.getenv("OPENAI_API_KEY") else "hash"
    return value


def openai_embedding_model() -> str:
    return os.getenv("DOCQA_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
