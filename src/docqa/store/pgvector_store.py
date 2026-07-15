"""Production vector store on Postgres + pgvector.

Search runs inside the database against an HNSW index (cosine distance), so it
scales past what brute force handles and inherits Postgres durability,
backups, and concurrent access. Start one locally with `docker compose up -d`.

The lexical leg of hybrid retrieval runs on Postgres full-text search: a
generated ``tsvector`` column with a GIN index, ranked by ``ts_rank_cd``.
"""

import numpy as np

from ..models import Chunk, Document, SearchResult
from .base import DIM_META_KEY, VectorStore, tokenize_query


class PgVectorStore(VectorStore):
    backend = "pgvector"

    def __init__(self, database_url: str):
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "Postgres support is not installed. "
                "Install it with: pip install 'mcp-docqa-server[pg]'"
            ) from exc
        self._conn = psycopg.connect(database_url, autocommit=True)
        # pgvector's adapter needs the extension to exist before it can register
        # the vector type with this connection.
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)

    def ensure_schema(self, dim: int) -> None:
        stored = self.get_meta(DIM_META_KEY)
        if stored is not None and int(stored) != dim:
            raise RuntimeError(
                f"Index stores {stored}-dim vectors but the current embedder "
                f"produces {dim}-dim vectors. Re-index with --force."
            )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
                doc_id       TEXT PRIMARY KEY,
                title        TEXT NOT NULL,
                url          TEXT NOT NULL,
                text         TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                n_chunks     INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          TEXT PRIMARY KEY,
                doc_id      TEXT NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text        TEXT NOT NULL,
                embedding   vector({dim}) NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
        # Lexical leg: generated tsvector column (auto-backfills existing rows).
        self._conn.execute(
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS text_tsv tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_text_tsv_gin ON chunks USING gin (text_tsv)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        if stored is None:
            self.set_meta(DIM_META_KEY, str(dim))

    def _meta_table_exists(self) -> bool:
        row = self._conn.execute("SELECT to_regclass('meta')").fetchone()
        return row is not None and row[0] is not None

    def get_meta(self, key: str) -> str | None:
        if not self._meta_table_exists():
            return None
        row = self._conn.execute("SELECT value FROM meta WHERE key = %s", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (key, value),
        )

    def doc_content_hash(self, doc_id: str) -> str | None:
        if not self._meta_table_exists():
            return None
        row = self._conn.execute(
            "SELECT content_hash FROM docs WHERE doc_id = %s", (doc_id,)
        ).fetchone()
        return row[0] if row else None

    def upsert_document(
        self,
        doc: Document,
        chunks: list[Chunk],
        vectors: np.ndarray,
        content_hash: str,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE doc_id = %s", (doc.id,))
            cur.execute(
                "INSERT INTO docs (doc_id, title, url, text, content_hash, n_chunks) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (doc_id) DO UPDATE SET title = EXCLUDED.title, "
                "url = EXCLUDED.url, text = EXCLUDED.text, "
                "content_hash = EXCLUDED.content_hash, n_chunks = EXCLUDED.n_chunks",
                (doc.id, doc.title, doc.url, doc.text, content_hash, len(chunks)),
            )
            cur.executemany(
                "INSERT INTO chunks (id, doc_id, chunk_index, text, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                [
                    (
                        f"{chunk.doc_id}:{chunk.chunk_index}",
                        chunk.doc_id,
                        chunk.chunk_index,
                        chunk.text,
                        vectors[i].astype(np.float32),
                    )
                    for i, chunk in enumerate(chunks)
                ],
            )

    def search(self, vector: np.ndarray, k: int) -> list[SearchResult]:
        rows = self._conn.execute(
            "SELECT c.doc_id, c.chunk_index, d.title, d.url, c.text, "
            "       1 - (c.embedding <=> %s) AS score "
            "FROM chunks c JOIN docs d ON d.doc_id = c.doc_id "
            "ORDER BY score DESC LIMIT %s",
            (vector.astype(np.float32), k),
        ).fetchall()
        return [
            SearchResult(
                doc_id=row[0],
                chunk_index=row[1],
                title=row[2],
                url=row[3],
                text=row[4],
                score=round(float(row[5]), 4),
            )
            for row in rows
        ]

    def lexical_search(self, query: str, k: int) -> list[SearchResult]:
        tokens = tokenize_query(query)
        if not tokens:
            return []
        # OR semantics; tokens are pure [a-z0-9] so the tsquery string is inert.
        tsquery = " | ".join(tokens)
        rows = self._conn.execute(
            "SELECT c.doc_id, c.chunk_index, d.title, d.url, c.text, "
            "       ts_rank_cd(c.text_tsv, q) AS score "
            "FROM chunks c JOIN docs d ON d.doc_id = c.doc_id, "
            "     to_tsquery('english', %s) q "
            "WHERE c.text_tsv @@ q ORDER BY score DESC LIMIT %s",
            (tsquery, k),
        ).fetchall()
        return [
            SearchResult(
                doc_id=row[0],
                chunk_index=row[1],
                title=row[2],
                url=row[3],
                text=row[4],
                score=round(float(row[5]), 4),
            )
            for row in rows
        ]

    def get_document(self, doc_id: str) -> Document | None:
        row = self._conn.execute(
            "SELECT doc_id, title, url, text FROM docs WHERE doc_id = %s", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return Document(id=row[0], title=row[1], url=row[2], text=row[3])

    def stats(self) -> dict:
        docs = self._conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "backend": self.backend,
            "documents": docs,
            "chunks": chunks,
            "embedder": self.get_meta("embedder_id"),
            "embedding_dim": self.get_meta(DIM_META_KEY),
        }

    def close(self) -> None:
        self._conn.close()
