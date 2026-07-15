"""Query-time retrieval: three modes over one store.

* ``vector``  — embed the query, cosine similarity over chunk embeddings.
* ``lexical`` — keyword match ranked by the backend's full-text scorer (BM25 /
  ts_rank). Catches exact terms embeddings can blur: identifiers, drug names,
  error codes.
* ``hybrid``  — both legs fused with Reciprocal Rank Fusion (the default).
  RRF scores each chunk by the sum of 1/(60 + rank) across the legs it appears
  in, so agreement between legs is rewarded without having to calibrate the
  legs' incomparable score scales against each other.
"""

from dataclasses import replace

from .embeddings import Embedder, get_embedder
from .models import SearchResult
from .store import EMBEDDER_META_KEY, VectorStore, get_store

MODES = ("hybrid", "vector", "lexical")
RRF_C = 60  # standard damping constant from the original RRF paper


def rrf_merge(legs: list[list[SearchResult]], k: int, c: int = RRF_C) -> list[SearchResult]:
    """Fuse ranked result lists into one top-k list by reciprocal rank."""
    fused: dict[str, tuple[float, SearchResult]] = {}
    for leg in legs:
        for rank, result in enumerate(leg, start=1):
            key = f"{result.doc_id}:{result.chunk_index}"
            gain = 1.0 / (c + rank)
            if key in fused:
                fused[key] = (fused[key][0] + gain, fused[key][1])
            else:
                fused[key] = (gain, result)
    ranked = sorted(fused.values(), key=lambda pair: pair[0], reverse=True)[:k]
    return [replace(result, score=round(score, 4)) for score, result in ranked]


class Retriever:
    """Binds one store to one embedder and answers queries against them.

    Refuses to run if the store was indexed by a different embedder — results
    would be silently meaningless otherwise (see ``VectorStore.guard_embedder``).
    """

    def __init__(self, store: VectorStore | None = None, embedder: Embedder | None = None):
        self.store = store or get_store()
        self.embedder = embedder or get_embedder()
        self.store.guard_embedder(self.embedder.id)

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[SearchResult]:
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r} (expected one of {MODES})")
        query = query.strip()
        if not query:
            return []
        if self.store.get_meta(EMBEDDER_META_KEY) is None:
            raise RuntimeError(
                "The index is empty — ingest a corpus first, e.g. "
                "`docqa-ingest index --sample` for the bundled demo corpus."
            )
        if mode == "lexical":
            return self.store.lexical_search(query, k)
        vector = self.embedder.embed([query])[0]
        if mode == "vector":
            return self.store.search(vector, k)
        # hybrid: pull a deeper candidate pool from each leg, then fuse.
        depth = min(max(k * 4, 20), 50)
        legs = [self.store.search(vector, depth), self.store.lexical_search(query, depth)]
        return rrf_merge(legs, k)
