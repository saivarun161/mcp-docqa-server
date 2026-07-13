"""Query-time retrieval: embed the query, search the store, return ranked chunks."""

from .embeddings import Embedder, get_embedder
from .models import SearchResult
from .store import EMBEDDER_META_KEY, VectorStore, get_store


class Retriever:
    """Binds one store to one embedder and answers queries against them.

    Refuses to run if the store was indexed by a different embedder — results
    would be silently meaningless otherwise (see ``VectorStore.guard_embedder``).
    """

    def __init__(self, store: VectorStore | None = None, embedder: Embedder | None = None):
        self.store = store or get_store()
        self.embedder = embedder or get_embedder()
        self.store.guard_embedder(self.embedder.id)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        query = query.strip()
        if not query:
            return []
        if self.store.get_meta(EMBEDDER_META_KEY) is None:
            raise RuntimeError(
                "The index is empty — ingest a corpus first, e.g. "
                "`docqa-ingest index --sample` for the bundled demo corpus."
            )
        vector = self.embedder.embed([query])[0]
        return self.store.search(vector, k)
