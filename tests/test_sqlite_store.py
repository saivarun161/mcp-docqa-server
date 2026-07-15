import sqlite3

import pytest

from docqa.embeddings import HashingEmbedder
from docqa.ingest.pipeline import index_documents
from docqa.store.sqlite_store import SQLiteVectorStore
from tests.conftest import DOCS
from tests.store_suite import run_store_battery


def test_sqlite_store_battery(store):
    run_store_battery(store)


def test_legacy_index_without_fts_is_backfilled_on_open(tmp_path):
    """An index built before hybrid search existed has no chunks_fts table.
    Re-opening it must create and backfill that table so lexical search works
    against previously-ingested data without a forced re-index."""
    path = str(tmp_path / "legacy.db")
    embedder = HashingEmbedder()

    # Build a normal index, then drop the FTS table to mimic the old on-disk shape.
    store = SQLiteVectorStore(path)
    index_documents(DOCS, store, embedder)
    store.close()
    raw = sqlite3.connect(path)
    raw.execute("DROP TABLE chunks_fts")
    raw.commit()
    raw.close()

    # Re-open: ensure_schema should rebuild the FTS index from existing chunks.
    reopened = SQLiteVectorStore(path)
    try:
        reopened.ensure_schema(embedder.dim)
        hits = reopened.lexical_search("espresso extraction pressure", k=3)
        assert hits and hits[0].doc_id == "doc-espresso"
    finally:
        reopened.close()


def test_fresh_db_has_no_meta(tmp_path):
    store = SQLiteVectorStore(str(tmp_path / "empty.db"))
    try:
        assert store.get_meta("embedder_id") is None
        assert store.doc_content_hash("anything") is None
    finally:
        store.close()


def test_dimension_mismatch_rejected(store):
    store.ensure_schema(512)
    with pytest.raises(RuntimeError, match="dim"):
        store.ensure_schema(1536)
