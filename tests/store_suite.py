"""One behavioral battery both store backends must pass.

Backend-specific test modules construct their store and hand it here, so the
SQLite and pgvector implementations are held to identical semantics.
"""

import numpy as np

from docqa.embeddings import HashingEmbedder
from docqa.ingest.pipeline import content_hash
from docqa.models import Document
from docqa.store.base import VectorStore
from tests.conftest import DOCS


def _index(store: VectorStore, embedder: HashingEmbedder, docs: list[Document]):
    from docqa.chunking import chunk_document

    store.ensure_schema(embedder.dim)
    for doc in docs:
        chunks = chunk_document(doc)
        vectors = embedder.embed([c.text for c in chunks])
        store.upsert_document(doc, chunks, vectors, content_hash(doc))


def run_store_battery(store: VectorStore):
    embedder = HashingEmbedder()
    _index(store, embedder, DOCS)

    # -- meta roundtrip ----------------------------------------------------
    store.set_meta("embedder_id", embedder.id)
    assert store.get_meta("embedder_id") == embedder.id
    store.set_meta("embedder_id", "overwritten")
    assert store.get_meta("embedder_id") == "overwritten"
    store.set_meta("embedder_id", embedder.id)
    assert store.get_meta("missing-key") is None

    # -- search ranks the on-topic document first --------------------------
    query = embedder.embed(["stage 2 hypertension blood pressure treatment"])[0]
    results = store.search(query, k=2)
    assert len(results) == 2
    assert results[0].doc_id == "doc-hypertension"
    assert results[0].score >= results[1].score
    assert results[0].title == "Hypertension basics"
    assert results[0].url == "local://doc-hypertension"
    assert "hypertension" in results[0].text.lower()

    # -- k larger than the corpus is fine ----------------------------------
    assert len(store.search(query, k=50)) == len(DOCS)  # one chunk per tiny doc

    # -- re-upsert replaces, never duplicates -------------------------------
    before = store.stats()["chunks"]
    _index(store, embedder, [DOCS[0]])
    assert store.stats()["chunks"] == before

    # -- content hash bookkeeping -------------------------------------------
    assert store.doc_content_hash("doc-kafka") == content_hash(DOCS[1])
    assert store.doc_content_hash("nope") is None

    # -- full document fetch -------------------------------------------------
    doc = store.get_document("doc-espresso")
    assert doc is not None and doc.title == "Espresso extraction"
    assert store.get_document("nope") is None

    # -- stats ----------------------------------------------------------------
    stats = store.stats()
    assert stats["documents"] == len(DOCS)
    assert stats["backend"] == store.backend
    assert stats["embedder"] == embedder.id

    # -- vector/chunk count mismatch is rejected ------------------------------
    from docqa.chunking import chunk_document

    doc = DOCS[0]
    chunks = chunk_document(doc)
    bad = np.zeros((len(chunks) + 1, embedder.dim), dtype=np.float32)
    try:
        store.upsert_document(doc, chunks, bad, "hash")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("mismatched vectors/chunks should raise ValueError")
