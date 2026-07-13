"""mcp-docqa-server — document Q&A retrieval exposed over the Model Context Protocol.

The package ships the retrieval half of a RAG pipeline as reusable infrastructure:
ingest a public document corpus, embed it into a vector store (SQLite or
Postgres + pgvector), and let any MCP host query it through typed tools.
"""

__version__ = "0.1.0"
