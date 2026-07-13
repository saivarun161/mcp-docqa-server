"""Fetch a public corpus of PubMed abstracts via the NCBI E-utilities API.

E-utilities is free and keyless (rate-limited to ~3 requests/second, which the
module respects). Only public abstracts are fetched — never index proprietary
or employer documents into a demo corpus.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterator

from ..models import Document

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "mcp-docqa-server"
REQUEST_DELAY_SECONDS = 0.4  # stay under NCBI's 3 req/s keyless limit
FETCH_BATCH_SIZE = 100


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": TOOL_NAME})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def search_pmids(query: str, max_docs: int) -> list[str]:
    """Return up to ``max_docs`` PubMed ids matching ``query``."""
    params = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "term": query,
            "retmax": max_docs,
            "retmode": "json",
            "sort": "relevance",
            "tool": TOOL_NAME,
        }
    )
    payload = json.loads(_get(f"{EUTILS_BASE}/esearch.fcgi?{params}"))
    return payload.get("esearchresult", {}).get("idlist", [])


def _abstract_text(article: ET.Element) -> str:
    """Join abstract sections, keeping labels like BACKGROUND:/RESULTS: as prefixes."""
    sections = []
    for node in article.findall(".//Abstract/AbstractText"):
        text = "".join(node.itertext()).strip()
        if not text:
            continue
        label = node.get("Label")
        sections.append(f"{label}: {text}" if label else text)
    return "\n".join(sections)


def fetch_abstracts(pmids: list[str]) -> Iterator[Document]:
    """Yield Documents for the given PubMed ids, skipping entries with no abstract."""
    for start in range(0, len(pmids), FETCH_BATCH_SIZE):
        if start > 0:
            time.sleep(REQUEST_DELAY_SECONDS)
        batch = pmids[start : start + FETCH_BATCH_SIZE]
        params = urllib.parse.urlencode(
            {
                "db": "pubmed",
                "id": ",".join(batch),
                "rettype": "abstract",
                "retmode": "xml",
                "tool": TOOL_NAME,
            }
        )
        root = ET.fromstring(_get(f"{EUTILS_BASE}/efetch.fcgi?{params}"))
        for article in root.findall(".//PubmedArticle"):
            pmid_node = article.find(".//PMID")
            title_node = article.find(".//ArticleTitle")
            if pmid_node is None or pmid_node.text is None:
                continue
            pmid = pmid_node.text.strip()
            title = "".join(title_node.itertext()).strip() if title_node is not None else ""
            abstract = _abstract_text(article)
            if not abstract:
                continue
            yield Document(
                id=f"pubmed-{pmid}",
                title=title or f"PubMed {pmid}",
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                text=abstract,
            )


def fetch_corpus(query: str, max_docs: int = 100) -> list[Document]:
    """Search PubMed for ``query`` and fetch the matching abstracts."""
    pmids = search_pmids(query, max_docs)
    time.sleep(REQUEST_DELAY_SECONDS)
    return list(fetch_abstracts(pmids))
