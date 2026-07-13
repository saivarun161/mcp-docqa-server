"""The indexing pipeline: chunk -> embed -> upsert, idempotently.

Each document's content hash is stored at index time; unchanged documents are
skipped on re-runs so repeated ingests don't re-embed (or re-bill) anything.
"""

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_document
from ..embeddings import Embedder
from ..models import Document
from ..store import EMBEDDER_META_KEY, VectorStore


@dataclass
class IndexReport:
    indexed: int = 0
    skipped: int = 0
    chunks: int = 0
    doc_ids: list[str] = field(default_factory=list)


def content_hash(doc: Document) -> str:
    payload = f"{doc.title}\x00{doc.url}\x00{doc.text}".encode()
    return hashlib.sha256(payload).hexdigest()


def load_corpus_jsonl(path: str | Path) -> list[Document]:
    """Load documents from JSONL with keys id (or doc_id), title, url, text."""
    docs = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = record.get("id") or record.get("doc_id")
            text = (record.get("text") or "").strip()
            if not doc_id or not text:
                raise ValueError(f"{path}:{line_no}: every record needs an id and text")
            docs.append(
                Document(
                    id=str(doc_id),
                    title=record.get("title", ""),
                    url=record.get("url", ""),
                    text=text,
                )
            )
    return docs


def index_documents(
    docs: Iterable[Document],
    store: VectorStore,
    embedder: Embedder,
    force: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    on_progress: Callable[[str], None] | None = None,
) -> IndexReport:
    """Index ``docs`` into ``store`` using ``embedder``.

    Skips documents whose content hash is already recorded (unless ``force``),
    and stamps the embedder id into store metadata so mismatched query-time
    configurations fail loudly instead of returning nonsense.
    """
    store.ensure_schema(embedder.dim)
    store.guard_embedder(embedder.id)

    report = IndexReport()
    for doc in docs:
        digest = content_hash(doc)
        if not force and store.doc_content_hash(doc.id) == digest:
            report.skipped += 1
            continue
        chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            report.skipped += 1
            continue
        vectors = embedder.embed([chunk.text for chunk in chunks])
        store.upsert_document(doc, chunks, vectors, digest)
        report.indexed += 1
        report.chunks += len(chunks)
        report.doc_ids.append(doc.id)
        if on_progress:
            on_progress(f"indexed {doc.id} ({len(chunks)} chunks) — {doc.title[:60]}")

    if report.indexed:
        store.set_meta(EMBEDDER_META_KEY, embedder.id)
    return report
