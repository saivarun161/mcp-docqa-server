# mcp-docqa-server

[![CI](https://github.com/saivarun161/mcp-docqa-server/actions/workflows/ci.yml/badge.svg)](https://github.com/saivarun161/mcp-docqa-server/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An [MCP](https://modelcontextprotocol.io) server that gives any AI client **semantic search over a document corpus** — the retrieval half of a RAG pipeline, shipped as reusable infrastructure. Point Claude Desktop (or any MCP host) at it and the model can search, read, and cite your documents autonomously; generation stays in the client, retrieval lives here.

```text
"What does the corpus say about the hour-1 sepsis bundle?"
        │
        ▼                      MCP (stdio / HTTP)
┌──────────────┐   search_documents("hour-1 sepsis bundle", k=5)   ┌───────────────┐
│  Claude /    │ ────────────────────────────────────────────────► │ docqa server  │
│  any MCP     │ ◄──────────────────────────────────────────────── │ embed → ANN   │
│  host        │     top-k chunks + titles, urls, scores           │ search → rank │
└──────────────┘                                                   └──────┬────────┘
                                                                          │
                                                          SQLite (embedded, exact)
                                                          or Postgres + pgvector (HNSW)
```

## Why this exists

Most RAG demos hard-wire retrieval into one chatbot. Exposing retrieval through MCP inverts that: **index once, query from anywhere** — Claude Desktop, an IDE agent, a CI job, your own client. The server is deliberately boring infrastructure: typed tools, two interchangeable storage backends, pluggable embeddings, an eval harness, and loud failures where silent ones usually live (see [Design decisions](#design-decisions)).

## Features

- **Four typed MCP tools** — `search_documents`, `fetch_document`, `corpus_stats`, `ping` — with docstrings written for the calling model, because tool descriptions *are* the interface.
- **Hybrid retrieval by default**: semantic (vector) and keyword (BM25 / Postgres FTS) search fused with Reciprocal Rank Fusion, so exact terms embeddings blur — identifiers, drug names, error codes — still land. Callers can force `vector` or `lexical` per query.
- **Two vector stores, one contract**: embedded **SQLite** (zero infrastructure, exact brute-force cosine + FTS5) and **Postgres + pgvector** (HNSW index + GIN full-text, production posture). Both pass the same behavioral test battery.
- **Pluggable embeddings**: OpenAI `text-embedding-3-small`, or a deterministic keyless hashing embedder so a fresh clone works with **no API key, no database, no network**.
- **Idempotent ingestion**: per-document content hashes mean re-runs skip unchanged docs — nothing gets re-embedded (or re-billed) by accident.
- **Corpus fetcher** for PubMed abstracts via the keyless NCBI E-utilities API (rate-limit aware).
- **Retrieval eval harness**: recall@k and MRR against a labeled testset, usable as a CI quality gate (`docqa-eval --min-recall5 0.9`).
- **stdio + Streamable HTTP transports**, Dockerfile included.
- **CI that means it**: lint, unit tests, a real MCP client round-trip over stdio, and an integration job against a live pgvector service container that ends by gating on retrieval recall.

## Quickstart — 60 seconds, no API key

```bash
git clone https://github.com/saivarun161/mcp-docqa-server.git
cd mcp-docqa-server
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

docqa-ingest index --sample   # bundle of 12 healthcare docs -> SQLite index
docqa-eval                    # recall@1/3/5 + MRR against the bundled testset
```

You now have a working index at `data/index.db`. Talk to it over real MCP with the Inspector:

```bash
npx @modelcontextprotocol/inspector .venv/bin/docqa-server
```

### Wire it into Claude Desktop

1. Open `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows).
2. Add the block from [`claude_desktop_config.example.json`](claude_desktop_config.example.json) with **absolute paths** — hosts launch servers from their own working directory, so relative paths break.
3. Fully restart Claude Desktop and ask: *"Search the docqa corpus: what counts as stage 2 hypertension?"*

The model will call `search_documents`, read the chunks, and answer with sources.

## A real corpus

Index a few hundred PubMed abstracts on any topic (keyless, public data):

```bash
docqa-ingest fetch --query "semaglutide cardiovascular outcomes" --max-docs 200
docqa-ingest index --corpus data/corpus.jsonl
docqa-ingest stats
```

Any JSONL with `id`, `title`, `url`, `text` fields works — swap PubMed for arXiv, EDGAR filings, or your own notes.

## Production posture

**Semantic embeddings** — put an OpenAI key in `.env` (see [`.env.example`](.env.example)) and re-index; `DOCQA_EMBEDDINGS=auto` picks it up:

```bash
pip install -e ".[openai]"
docqa-ingest index --corpus data/corpus.jsonl --force
```

**Postgres + pgvector** — vectors move into an HNSW-indexed table; search runs inside the database:

```bash
docker compose up -d        # pgvector/pgvector:pg16 with the extension enabled
export DOCQA_STORE=pgvector DATABASE_URL=postgresql://docqa:docqa@localhost:5432/docqa
pip install -e ".[pg]"
docqa-ingest index --sample
```

**HTTP transport** — for network-reachable deployments instead of stdio:

```bash
docqa-server --transport http --host 0.0.0.0 --port 8000
# or containerized:
docker build -t mcp-docqa-server . && docker run -p 8000:8000 mcp-docqa-server
```

## MCP tools

| Tool | Arguments | Returns |
|---|---|---|
| `search_documents` | `query`, `k=5`, `mode="hybrid"` | top-k chunks with `doc_id`, `title`, `url`, `text`, `score` |
| `fetch_document` | `doc_id` | the full source document |
| `corpus_stats` | — | doc/chunk counts, backend, embedder that built the index |
| `ping` | — | `"pong"` (connectivity check) |

`mode` selects the retrieval strategy: `hybrid` (default) fuses both legs, `vector` is semantic-only (best for paraphrased/conceptual questions), `lexical` is keyword-only (best when an exact term must appear).

## Retrieval quality

`docqa-eval` retrieves for every testset question and reports where the expected document ranked:

```text
Retrieval eval — 12 questions, k=5, mode=hybrid
store=sqlite  embedder=hash-v1-512

  [rank 1] What blood pressure reading counts as stage 2 hypertension?  (expects sample-001)
  ...
recall@1=1.00  recall@3=1.00  recall@5=1.00  MRR=1.00
```

Pass `--mode vector|lexical|hybrid` to compare retrieval strategies on the same testset. The bundled corpus is small and topically distinct, so every mode scores perfectly — that run proves the *plumbing*. The interesting experiments start when you index a few hundred PubMed abstracts and compare `hash` vs `openai` embeddings, or `vector` vs `hybrid`, on your own testset; CI runs the eval against a live pgvector container and fails the build if recall@5 drops below 0.9.

## Design decisions

- **Hybrid retrieval fuses with RRF, not score-mixing.** Vector cosine and BM25 live on incomparable scales, so blending their raw scores needs fragile per-corpus tuning. Reciprocal Rank Fusion instead combines *ranks* — each chunk scores `Σ 1/(60 + rank)` over the legs it appears in — which needs no calibration and rewards chunks both legs agree on. Each leg runs in its own engine (NumPy cosine / SQLite FTS5 / Postgres GIN); the retriever pulls a deeper candidate pool from each, then fuses.
- **Embedder identity is persisted and enforced.** Vectors from different embedders live in unrelated spaces; querying an OpenAI-built index with hash vectors doesn't error mathematically — it just returns garbage. The store records which embedder built it and the retriever refuses a mismatch with an actionable message. Silent failure → loud failure.
- **Brute force is a feature at SQLite scale.** Exact cosine over a few thousand chunks is milliseconds with NumPy and has zero recall loss; ANN indexes buy speed at scale, not correctness. The pgvector backend adds HNSW when the corpus outgrows brute force.
- **Chunks carry their title.** Each chunk is prefixed with its document title before embedding, so a chunk ripped out of context still knows what it's about.
- **One behavioral battery, two backends.** The SQLite and pgvector stores pass the identical test suite (`tests/store_suite.py`), which is what "interchangeable" actually means.
- **The MCP layer is tested with a real MCP client.** CI spawns the server over stdio and drives it with an `mcp.ClientSession` — the same handshake Claude Desktop performs — not by calling Python functions directly.
- **Public data only.** PubMed abstracts and original sample docs. Never index proprietary or employer documents into a demo corpus.

## Project structure

```text
src/docqa/
├── server.py            # FastMCP server + tool definitions
├── retriever.py         # hybrid/vector/lexical modes + RRF fusion, embedder guard
├── embeddings.py        # OpenAIEmbedder | HashingEmbedder (keyless fallback)
├── chunking.py          # word windows with overlap
├── config.py            # env-driven settings (.env aware)
├── store/
│   ├── base.py          # VectorStore contract (vector + lexical) + embedder guard
│   ├── sqlite_store.py  # embedded: brute-force cosine + FTS5 BM25
│   └── pgvector_store.py# Postgres: pgvector HNSW + GIN full-text
├── ingest/
│   ├── pubmed.py        # NCBI E-utilities fetcher (keyless, rate-limited)
│   ├── pipeline.py      # chunk -> embed -> upsert, content-hash idempotent
│   └── cli.py           # docqa-ingest fetch | index | stats
├── eval/run_eval.py     # recall@k + MRR, CI-gateable
└── data/                # bundled 12-doc sample corpus + labeled testset
tests/                   # unit + store battery + MCP stdio round-trip
.github/workflows/ci.yml # lint, tests, live pgvector integration + recall gate
```

## Roadmap

- [x] MCP tools over stdio + Streamable HTTP
- [x] SQLite and pgvector backends behind one contract
- [x] Idempotent ingestion + PubMed fetcher
- [x] Eval harness with CI recall gate
- [x] Hybrid retrieval (BM25 + vector, reciprocal rank fusion)
- [ ] Cross-encoder reranking stage
- [ ] More corpus adapters (arXiv, EDGAR)
- [ ] Bearer-token auth for the HTTP transport

## License

MIT — see [LICENSE](LICENSE).

Built by [Varun Kammadanam](https://www.linkedin.com/in/varun-kammadanam-a823a6196) — backend + GenAI engineer (Java, Python, AWS, RAG systems).
