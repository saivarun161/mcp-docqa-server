"""Embedded vector store on SQLite + NumPy.

Zero infrastructure: vectors are float32 blobs in SQLite and similarity is a
brute-force normalized dot product. That is exact (no ANN approximation) and
entirely adequate for corpora in the tens of thousands of chunks; beyond that,
switch to the pgvector backend, which searches an HNSW index instead.

The lexical leg of hybrid retrieval runs on SQLite's built-in FTS5 engine
(BM25 ranking) — still zero infrastructure.
"""

import sqlite3
from pathlib import Path

import numpy as np

from ..models import Chunk, Document, SearchResult
from .base import DIM_META_KEY, VectorStore, tokenize_query

_SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    doc_id       TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    url          TEXT NOT NULL,
    text         TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    n_chunks     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SQLiteVectorStore(VectorStore):
    backend = "sqlite"

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")

    def ensure_schema(self, dim: int) -> None:
        with self._conn:
            fts_existed = self._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'chunks_fts'"
            ).fetchone()
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                "USING fts5(id UNINDEXED, doc_id UNINDEXED, text)"
            )
            if not fts_existed:
                # Migrate indexes built before the lexical leg existed.
                self._conn.execute(
                    "INSERT INTO chunks_fts (id, doc_id, text) SELECT id, doc_id, text FROM chunks"
                )
            stored = self.get_meta(DIM_META_KEY)
            if stored is None:
                self.set_meta(DIM_META_KEY, str(dim))
            elif int(stored) != dim:
                raise RuntimeError(
                    f"Index at {self.path} stores {stored}-dim vectors but the current "
                    f"embedder produces {dim}-dim vectors. Re-index with --force."
                )

    def get_meta(self, key: str) -> str | None:
        try:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        except sqlite3.OperationalError:  # meta table not created yet
            return None
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def doc_content_hash(self, doc_id: str) -> str | None:
        try:
            row = self._conn.execute(
                "SELECT content_hash FROM docs WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            return None
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
        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc.id,))
            self._conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc.id,))
            self._conn.execute(
                "INSERT INTO docs (doc_id, title, url, text, content_hash, n_chunks) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(doc_id) DO UPDATE SET title = excluded.title, "
                "url = excluded.url, text = excluded.text, "
                "content_hash = excluded.content_hash, n_chunks = excluded.n_chunks",
                (doc.id, doc.title, doc.url, doc.text, content_hash, len(chunks)),
            )
            self._conn.executemany(
                "INSERT INTO chunks (id, doc_id, chunk_index, text, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        f"{chunk.doc_id}:{chunk.chunk_index}",
                        chunk.doc_id,
                        chunk.chunk_index,
                        chunk.text,
                        vectors[i].astype(np.float32).tobytes(),
                    )
                    for i, chunk in enumerate(chunks)
                ],
            )
            self._conn.executemany(
                "INSERT INTO chunks_fts (id, doc_id, text) VALUES (?, ?, ?)",
                [
                    (f"{chunk.doc_id}:{chunk.chunk_index}", chunk.doc_id, chunk.text)
                    for chunk in chunks
                ],
            )

    def search(self, vector: np.ndarray, k: int) -> list[SearchResult]:
        rows = self._conn.execute(
            "SELECT c.doc_id, c.chunk_index, c.text, c.embedding, d.title, d.url "
            "FROM chunks c JOIN docs d ON d.doc_id = c.doc_id"
        ).fetchall()
        if not rows:
            return []
        matrix = np.frombuffer(b"".join(row[3] for row in rows), dtype=np.float32).reshape(
            len(rows), -1
        )
        scores = matrix @ vector.astype(np.float32)
        k = min(k, len(rows))
        top = np.argsort(scores)[::-1][:k]
        return [
            SearchResult(
                doc_id=rows[i][0],
                chunk_index=rows[i][1],
                title=rows[i][4],
                url=rows[i][5],
                text=rows[i][2],
                score=round(float(scores[i]), 4),
            )
            for i in top
        ]

    def lexical_search(self, query: str, k: int) -> list[SearchResult]:
        tokens = tokenize_query(query)
        if not tokens:
            return []
        # OR semantics over quoted tokens; quoting keeps FTS5 operators inert.
        match = " OR ".join(f'"{token}"' for token in tokens)
        rows = self._conn.execute(
            "SELECT f.id, bm25(chunks_fts) AS rank FROM chunks_fts f "
            "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (match, k),
        ).fetchall()
        results = []
        for chunk_id, rank in rows:
            row = self._conn.execute(
                "SELECT c.doc_id, c.chunk_index, c.text, d.title, d.url "
                "FROM chunks c JOIN docs d ON d.doc_id = c.doc_id WHERE c.id = ?",
                (chunk_id,),
            ).fetchone()
            if row is None:  # FTS row orphaned by a concurrent delete
                continue
            results.append(
                SearchResult(
                    doc_id=row[0],
                    chunk_index=row[1],
                    title=row[3],
                    url=row[4],
                    text=row[2],
                    score=round(-float(rank), 4),  # bm25(): lower is better; flip sign
                )
            )
        return results

    def get_document(self, doc_id: str) -> Document | None:
        row = self._conn.execute(
            "SELECT doc_id, title, url, text FROM docs WHERE doc_id = ?", (doc_id,)
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
