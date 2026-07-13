"""The MCP server: document retrieval exposed as typed, discoverable tools.

Tool docstrings ARE the interface — the calling model reads them to decide
when and how to use each tool, so they are written for the model, not for
humans browsing the code.

Run locally (stdio, what Claude Desktop uses):
    docqa-server

Run as an HTTP service (Streamable HTTP transport):
    docqa-server --transport http --host 0.0.0.0 --port 8000
"""

import argparse
from functools import lru_cache

from mcp.server.fastmcp import FastMCP

from .retriever import Retriever

mcp = FastMCP("docqa")

MAX_K = 25


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    """One store connection + embedder per process, created on first tool call."""
    return Retriever()


@mcp.tool()
def ping() -> str:
    """Health check. Returns 'pong' to confirm the docqa server is reachable.

    Use this to verify the connection between the host and this server.
    """
    return "pong"


@mcp.tool()
def search_documents(query: str, k: int = 5) -> list[dict]:
    """Semantic search over the indexed document corpus.

    Returns the k best-matching text chunks, each with its source metadata:
    doc_id, chunk_index, title, url, the chunk text, and a relevance score in
    [0, 1] (higher is better). Results may include multiple chunks from the
    same document. Use fetch_document with a result's doc_id to read the full
    source document.

    Args:
        query: A natural-language question or search phrase.
        k: How many chunks to return (default 5, max 25).
    """
    k = max(1, min(int(k), MAX_K))
    return [result.to_dict() for result in _retriever().search(query, k)]


@mcp.tool()
def fetch_document(doc_id: str) -> dict:
    """Fetch one full source document by its id.

    Use this after search_documents to read a promising source in full instead
    of reasoning from a chunk. Returns id, title, url, and the complete text.

    Args:
        doc_id: The document id exactly as returned by search_documents.
    """
    doc = _retriever().store.get_document(doc_id.strip())
    if doc is None:
        raise ValueError(
            f"No document with id '{doc_id}'. Use a doc_id returned by search_documents."
        )
    return {"id": doc.id, "title": doc.title, "url": doc.url, "text": doc.text}


@mcp.tool()
def corpus_stats() -> dict:
    """Describe the indexed corpus: document/chunk counts, storage backend, and
    which embedding model built the index.

    Call this first if you are unsure whether the corpus is relevant to the
    user's question or whether anything has been indexed at all.
    """
    return _retriever().store.stats()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the docqa MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="stdio for local hosts like Claude Desktop (default); "
        "http for a network-reachable Streamable HTTP server",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP bind port")
    args = parser.parse_args()

    if args.transport == "http":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
