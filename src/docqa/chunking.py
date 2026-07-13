"""Word-window chunking with overlap.

Splitting on words rather than tokens keeps the module dependency-free and
deterministic across platforms; 200 words approximates the ~250-300 tokens that
works well for abstract-sized documents. Overlap preserves context across
chunk boundaries so a sentence straddling two windows is retrievable from both.
"""

from .models import Chunk, Document

DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 40


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split ``text`` into word windows of ``chunk_size`` with ``overlap`` words shared
    between consecutive windows. Returns [] for empty/whitespace input."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def chunk_document(
    doc: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk a document, prefixing the title so every chunk carries its context."""
    body = f"{doc.title}\n\n{doc.text}" if doc.title else doc.text
    return [
        Chunk(doc_id=doc.id, chunk_index=i, text=piece)
        for i, piece in enumerate(chunk_text(body, chunk_size, overlap))
    ]
