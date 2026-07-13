import json

import pytest

from docqa.embeddings import HashingEmbedder
from docqa.ingest.pipeline import index_documents, load_corpus_jsonl
from docqa.sampledata import sample_corpus_path, sample_testset_path


def test_index_then_skip_then_force(store, embedder, docs):
    first = index_documents(docs, store, embedder)
    assert first.indexed == len(docs)
    assert first.skipped == 0
    assert first.chunks >= len(docs)

    again = index_documents(docs, store, embedder)
    assert again.indexed == 0
    assert again.skipped == len(docs)

    forced = index_documents(docs, store, embedder, force=True)
    assert forced.indexed == len(docs)
    assert store.stats()["documents"] == len(docs)


def test_changed_document_is_reindexed(store, embedder, docs):
    index_documents(docs, store, embedder)
    changed = docs[0].__class__(
        id=docs[0].id, title=docs[0].title, url=docs[0].url, text=docs[0].text + " updated."
    )
    report = index_documents([changed], store, embedder)
    assert report.indexed == 1
    assert store.get_document(docs[0].id).text.endswith("updated.")


def test_embedder_mismatch_is_rejected(store, embedder, docs):
    index_documents(docs, store, embedder)

    class OtherEmbedder(HashingEmbedder):
        def __init__(self):
            super().__init__()
            self.id = "hash-v2-different"

    with pytest.raises(RuntimeError, match="embedder"):
        index_documents(docs, store, OtherEmbedder())


def test_load_corpus_jsonl_validates(tmp_path):
    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps({"id": "a", "title": "T", "url": "u", "text": "some text"})
        + "\n\n"
        + json.dumps({"doc_id": "b", "text": "doc_id alias works"})
        + "\n"
    )
    docs = load_corpus_jsonl(good)
    assert [d.id for d in docs] == ["a", "b"]

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "a", "text": "  "}) + "\n")
    with pytest.raises(ValueError, match="id and text"):
        load_corpus_jsonl(bad)


def test_bundled_sample_corpus_and_testset_are_consistent():
    with sample_corpus_path() as path:
        docs = load_corpus_jsonl(path)
    assert len(docs) == 12
    doc_ids = {d.id for d in docs}

    with sample_testset_path() as path:
        cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(cases) >= 10
    # every expected answer must actually exist in the corpus
    assert {c["expected_doc_id"] for c in cases} <= doc_ids
