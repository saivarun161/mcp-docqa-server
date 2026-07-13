"""Core value types passed between ingestion, storage, retrieval, and the server."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Document:
    """A source document as fetched from a corpus (one PubMed abstract, etc.)."""

    id: str
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of one document, the unit that gets embedded."""

    doc_id: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk plus its source metadata and similarity score."""

    doc_id: str
    chunk_index: int
    title: str
    url: str
    text: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)
