"""Pluggable text embedders.

Two implementations behind one interface:

* ``OpenAIEmbedder`` — semantic embeddings via the OpenAI API (production path).
* ``HashingEmbedder`` — deterministic lexical feature-hashing. Not semantic; it
  exists so tests, CI, and a fresh clone can exercise the entire pipeline with
  zero credentials and zero network access.

Every embedder returns L2-normalized float32 rows, so cosine similarity is a
plain dot product everywhere downstream. The embedder ``id`` is persisted in
the store's metadata: an index built with one embedder refuses queries from
another, because mixing vector spaces silently returns garbage.
"""

import hashlib
import itertools
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from . import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Anything that can turn a batch of texts into (n, dim) unit vectors."""

    id: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Feature-hashing bag of unigrams + bigrams, L2-normalized.

    Uses md5 (not Python's salted ``hash()``) so vectors are identical across
    processes, platforms, and runs — a hard requirement for a persisted index.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.id = f"hash-v1-{dim}"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = _TOKEN_RE.findall(text.lower())
        bigrams = [f"{a}_{b}" for a, b in itertools.pairwise(words)]
        return words + bigrams

    def _bucket(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                matrix[row, self._bucket(token)] += 1.0
        return _normalize(matrix)


class OpenAIEmbedder:
    """Semantic embeddings via the OpenAI embeddings API (batched)."""

    BATCH_SIZE = 128

    def __init__(self, model: str | None = None, dim: int = 1536):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "The 'openai' package is not installed. "
                "Install it with: pip install 'mcp-docqa-server[openai]'"
            ) from exc
        self._client = OpenAI()
        self.model = model or config.openai_embedding_model()
        self.dim = dim
        self.id = f"openai:{self.model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = list(texts[start : start + self.BATCH_SIZE])
            response = self._client.embeddings.create(model=self.model, input=batch)
            rows.extend(item.embedding for item in response.data)
        return _normalize(np.asarray(rows, dtype=np.float32))


def get_embedder(name: str | None = None) -> Embedder:
    """Build the configured embedder. ``name`` overrides the environment."""
    backend = (name or config.embeddings_backend()).strip().lower()
    if backend == "hash":
        return HashingEmbedder()
    if backend == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"Unknown embeddings backend {backend!r} (expected 'openai' or 'hash')")
