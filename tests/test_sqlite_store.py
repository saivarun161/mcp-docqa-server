import pytest

from docqa.store.sqlite_store import SQLiteVectorStore
from tests.store_suite import run_store_battery


def test_sqlite_store_battery(store):
    run_store_battery(store)


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
