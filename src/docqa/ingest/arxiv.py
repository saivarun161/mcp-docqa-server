"""Fetch a public corpus of arXiv paper abstracts via the arXiv Atom API.

The arXiv API is free and keyless. arXiv asks clients to allow a few seconds
between requests and to identify themselves, both of which this module does.
Only public abstracts are fetched — never index proprietary documents.

The Atom parser is factored out of the network call so it can be unit-tested
against a fixture with no HTTP.

API docs: https://info.arxiv.org/help/api/user-manual.html
"""

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from ..models import Document

ARXIV_API = "https://export.arxiv.org/api/query"
TOOL_NAME = "mcp-docqa-server"
REQUEST_DELAY_SECONDS = 3.0  # arXiv asks for ~3s between programmatic requests
_ATOM = {"atom": "http://www.w3.org/2005/Atom"}
_VERSION_SUFFIX = re.compile(r"v\d+$")


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": TOOL_NAME})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _normalize(text: str | None) -> str:
    """Collapse the newlines and runs of spaces arXiv wraps titles/abstracts in."""
    return " ".join(text.split()) if text else ""


def _arxiv_id(id_url: str) -> str:
    """Extract the bare arXiv id from an entry id URL, dropping the version.

    'http://arxiv.org/abs/2301.01234v2' -> '2301.01234'
    'http://arxiv.org/abs/cs/0301001v1' -> 'cs/0301001'
    """
    tail = id_url.rsplit("/abs/", 1)[-1]
    return _VERSION_SUFFIX.sub("", tail)


def parse_atom(xml_bytes: bytes) -> list[Document]:
    """Parse an arXiv Atom response into Documents, skipping entries with no abstract."""
    root = ET.fromstring(xml_bytes)
    docs: list[Document] = []
    for entry in root.findall("atom:entry", _ATOM):
        id_node = entry.find("atom:id", _ATOM)
        summary_node = entry.find("atom:summary", _ATOM)
        if id_node is None or id_node.text is None:
            continue
        abstract = _normalize(summary_node.text if summary_node is not None else "")
        if not abstract:
            continue
        aid = _arxiv_id(id_node.text.strip())
        title = _normalize(entry.findtext("atom:title", default="", namespaces=_ATOM))
        docs.append(
            Document(
                id=f"arxiv-{aid}",
                title=title or f"arXiv {aid}",
                url=f"https://arxiv.org/abs/{aid}",
                text=abstract,
            )
        )
    return docs


def fetch_corpus(query: str, max_docs: int = 100) -> list[Document]:
    """Search arXiv for ``query`` and fetch the matching abstracts."""
    params = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_docs,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    docs = parse_atom(_get(f"{ARXIV_API}?{params}"))
    time.sleep(REQUEST_DELAY_SECONDS)  # be a polite API citizen before any follow-up
    return docs
