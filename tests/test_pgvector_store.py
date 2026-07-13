"""pgvector integration tests.

Skipped unless DATABASE_URL points at a running Postgres with the pgvector
extension available (locally: `docker compose up -d`; in CI: a service
container). The battery is the same one SQLite passes.
"""

import os

import pytest

pytestmark = pytest.mark.pgvector

DATABASE_URL = os.getenv("DATABASE_URL")


@pytest.fixture
def pg_store():
    if not DATABASE_URL:
        pytest.skip("DATABASE_URL not set; start Postgres with `docker compose up -d`")
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, docs, meta CASCADE")

    from docqa.store.pgvector_store import PgVectorStore

    store = PgVectorStore(DATABASE_URL)
    yield store
    store.close()


def test_pgvector_store_battery(pg_store):
    from tests.store_suite import run_store_battery

    run_store_battery(pg_store)


def test_pgvector_dimension_mismatch_rejected(pg_store):
    pg_store.ensure_schema(512)
    with pytest.raises(RuntimeError, match="dim"):
        pg_store.ensure_schema(1536)
