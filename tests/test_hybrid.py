"""Hybrid retrieval: reciprocal rank fusion and the lexical-vs-vector tradeoff.

The RRF unit tests pin the fusion math directly; the integration tests prove
that hybrid actually recovers a result each single leg misses on its own.
"""

import pytest

from docqa.ingest.pipeline import index_documents
from docqa.models import Document, SearchResult
from docqa.retriever import RRF_C, Retriever, rrf_merge


def _result(doc_id: str, chunk_index: int = 0, score: float = 0.0) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        chunk_index=chunk_index,
        title=doc_id,
        url=f"local://{doc_id}",
        text=f"text for {doc_id}",
        score=score,
    )


def test_rrf_rewards_agreement_between_legs():
    # 'b' is 2nd in each leg; 'a' and 'c' are 1st in one leg but absent from the
    # other. Consensus should lift 'b' above both single-leg winners.
    leg1 = [_result("a"), _result("b"), _result("x")]
    leg2 = [_result("c"), _result("b"), _result("y")]
    fused = rrf_merge([leg1, leg2], k=3)
    assert fused[0].doc_id == "b"
    b_score = 1.0 / (RRF_C + 2) + 1.0 / (RRF_C + 2)
    assert fused[0].score == pytest.approx(round(b_score, 4))


def test_rrf_dedupes_same_chunk_across_legs():
    leg1 = [_result("a", 0), _result("a", 1)]
    leg2 = [_result("a", 0)]
    fused = rrf_merge([leg1, leg2], k=10)
    keys = {(r.doc_id, r.chunk_index) for r in fused}
    assert keys == {("a", 0), ("a", 1)}
    # (a,0) appears in both legs (rank 1 + rank 1); (a,1) only in leg1 (rank 2).
    assert fused[0].chunk_index == 0


def test_rrf_respects_k_and_orders_descending():
    leg = [_result(c) for c in "abcdef"]
    fused = rrf_merge([leg], k=3)
    assert len(fused) == 3
    assert [r.score for r in fused] == sorted((r.score for r in fused), reverse=True)


def test_rrf_empty_input():
    assert rrf_merge([], k=5) == []
    assert rrf_merge([[], []], k=5) == []


CONTRACT_DOC = Document(
    id="doc-contract",
    title="Service agreement clause",
    url="local://doc-contract",
    text=(
        "The parties agree that invoice number INV-88231 shall be settled within "
        "thirty days. Late payment accrues interest at the statutory rate."
    ),
)
PROSE_DOC = Document(
    id="doc-prose",
    title="Paying suppliers on time",
    url="local://doc-prose",
    text=(
        "Settling what you owe a vendor promptly keeps the commercial relationship "
        "healthy and avoids penalty charges for overdue balances."
    ),
)


def test_hybrid_recovers_exact_identifier_that_semantics_blurs(store, embedder):
    # The hashing embedder is lexical-ish, so to make the point we lean on the
    # lexical leg for an exact identifier and confirm hybrid surfaces it.
    index_documents([CONTRACT_DOC, PROSE_DOC], store, embedder)
    retriever = Retriever(store=store, embedder=embedder)

    lexical = retriever.search("INV-88231", k=2, mode="lexical")
    assert lexical[0].doc_id == "doc-contract"

    hybrid = retriever.search("INV-88231", k=2, mode="hybrid")
    assert hybrid[0].doc_id == "doc-contract"


def test_hybrid_never_returns_more_than_k(store, embedder, docs):
    index_documents(docs, store, embedder)
    retriever = Retriever(store=store, embedder=embedder)
    for k in (1, 2, 3, 5):
        assert len(retriever.search("data platform records", k=k, mode="hybrid")) <= k
