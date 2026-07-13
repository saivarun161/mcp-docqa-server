"""End-to-end MCP test: spawn the real server over stdio and talk to it with a
real MCP client session — exactly what Claude Desktop does, minus the model."""

import json
import os
import sys

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from docqa.embeddings import HashingEmbedder
from docqa.ingest.pipeline import index_documents
from docqa.sampledata import sample_corpus_path
from docqa.store.sqlite_store import SQLiteVectorStore

pytestmark = pytest.mark.anyio


@pytest.fixture
def indexed_db(tmp_path):
    """A real SQLite index of the bundled sample corpus, built with hash embeddings."""
    from docqa.ingest.pipeline import load_corpus_jsonl

    db_path = str(tmp_path / "index.db")
    store = SQLiteVectorStore(db_path)
    with sample_corpus_path() as corpus:
        index_documents(load_corpus_jsonl(corpus), store, HashingEmbedder())
    store.close()
    return db_path


def _server_params(indexed_db: str) -> StdioServerParameters:
    env = dict(os.environ)
    env.update(
        {
            "DOCQA_STORE": "sqlite",
            "DOCQA_SQLITE_PATH": indexed_db,
            "DOCQA_EMBEDDINGS": "hash",
        }
    )
    return StdioServerParameters(command=sys.executable, args=["-m", "docqa.server"], env=env)


def _text_of(result) -> str:
    return "\n".join(block.text for block in result.content if hasattr(block, "text"))


async def test_full_mcp_round_trip(indexed_db):
    with anyio.fail_after(60):
        async with (
            stdio_client(_server_params(indexed_db)) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()

            tools = {tool.name for tool in (await session.list_tools()).tools}
            assert tools == {"ping", "search_documents", "fetch_document", "corpus_stats"}

            pong = await session.call_tool("ping", {})
            assert not pong.isError
            assert _text_of(pong) == "pong"

            stats = await session.call_tool("corpus_stats", {})
            assert not stats.isError
            assert '"documents": 12' in _text_of(stats).replace("'", '"') or "12" in _text_of(stats)

            found = await session.call_tool(
                "search_documents",
                {"query": "stage 2 hypertension blood pressure threshold", "k": 3},
            )
            assert not found.isError
            assert "sample-001" in _text_of(found)

            doc = await session.call_tool("fetch_document", {"doc_id": "sample-001"})
            assert not doc.isError
            payload = json.loads(_text_of(doc))
            assert payload["title"].startswith("Hypertension")
            assert "140/90" in payload["text"]


async def test_unknown_document_is_a_tool_error(indexed_db):
    with anyio.fail_after(60):
        async with (
            stdio_client(_server_params(indexed_db)) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool("fetch_document", {"doc_id": "does-not-exist"})
            assert result.isError
            assert "does-not-exist" in _text_of(result)
