import pytest

from docqa.ingest.pipeline import index_documents
from docqa.retriever import Retriever


def test_search_returns_the_on_topic_document_first(store, embedder, docs):
    index_documents(docs, store, embedder)
    retriever = Retriever(store=store, embedder=embedder)

    results = retriever.search("how does kafka event streaming work", k=2)
    assert results[0].doc_id == "doc-kafka"
    assert len(results) == 2

    results = retriever.search("espresso grind size and extraction", k=1)
    assert results[0].doc_id == "doc-espresso"


def test_blank_query_returns_nothing(store, embedder, docs):
    index_documents(docs, store, embedder)
    retriever = Retriever(store=store, embedder=embedder)
    assert retriever.search("   ") == []


def test_empty_index_raises_actionable_error(store, embedder):
    retriever = Retriever(store=store, embedder=embedder)
    with pytest.raises(RuntimeError, match="ingest"):
        retriever.search("anything")


def test_mismatched_embedder_rejected_at_construction(store, embedder, docs):
    index_documents(docs, store, embedder)

    class OtherEmbedder(type(embedder)):
        def __init__(self):
            super().__init__()
            self.id = "hash-v9-elsewhere"

    with pytest.raises(RuntimeError, match="embedder"):
        Retriever(store=store, embedder=OtherEmbedder())
