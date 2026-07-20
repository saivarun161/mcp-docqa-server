"""Fetch a public corpus from any generic RSS 2.0 or Atom 1.0 feed.

Unlike the PubMed and arXiv adapters, which speak one site's bespoke API, this
adapter reads the two syndication formats the open web actually ships: RSS 2.0
(``<rss><channel><item>``) and Atom 1.0 (``<feed><entry>``). Point it at a blog,
a docs changelog, a journal's article feed, or an EDGAR Atom feed and it turns
each entry into a Document. Only fetch public feeds — never index proprietary
content into a demo corpus.

Feed bodies are usually HTML, so entry summaries are run through a small tag
stripper before they become chunk text. The parser is factored out of the
network call so it can be unit-tested against fixtures with no HTTP.

Format references:
  RSS 2.0  — https://www.rssboard.org/rss-specification
  Atom 1.0 — https://datatracker.ietf.org/doc/html/rfc4287
"""

import hashlib
import time
import urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

from ..models import Document

TOOL_NAME = "mcp-docqa-server"
REQUEST_DELAY_SECONDS = 1.0  # polite pause before any follow-up request
_ATOM_NS = "http://www.w3.org/2005/Atom"
_CONTENT_NS = "http://purl.org/rss/1.0/modules/content/"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": TOOL_NAME})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


class _TextExtractor(HTMLParser):
    """Collect the visible text of an HTML fragment, dropping tags and entities."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _strip_html(raw: str | None) -> str:
    """Turn a (possibly HTML) feed body into normalized plain text.

    Feeds routinely wrap summaries in markup and HTML entities; this yields the
    same collapsed-whitespace plain text the other adapters emit.
    """
    if not raw:
        return ""
    parser = _TextExtractor()
    parser.feed(raw)
    return " ".join(parser.text().split())


def _local(tag: str) -> str:
    """Return an element tag without its ``{namespace}`` prefix."""
    return tag.rsplit("}", 1)[-1]


def _doc_id(basis: str) -> str:
    """Derive a stable, filesystem-safe document id from a guid or link.

    Feed guids are arbitrary strings (URLs, URNs, opaque tokens), so we hash the
    identifying field rather than embed it raw. The same entry always maps to the
    same id, which is what the idempotent ingestion pipeline relies on.
    """
    # sha1 here is a content-addressing hash for a stable id, not a security primitive.
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()
    return f"feed-{digest[:16]}"


def _atom_link(entry: ET.Element) -> str:
    """Pick the best href from an Atom entry's ``<link>`` elements.

    Atom allows several links distinguished by ``rel``; the readable page is the
    ``alternate`` link (the default when ``rel`` is absent), so prefer it and
    fall back to the first link with an href.
    """
    fallback = ""
    for link in entry.findall(f"{{{_ATOM_NS}}}link"):
        href = link.get("href", "")
        if not href:
            continue
        if link.get("rel", "alternate") == "alternate":
            return href
        fallback = fallback or href
    return fallback


def _parse_rss(root: ET.Element) -> list[Document]:
    docs: list[Document] = []
    for item in root.iter("item"):
        title = _strip_html(item.findtext("title"))
        link = (item.findtext("link") or "").strip()
        guid = (item.findtext("guid") or "").strip()
        # content:encoded carries the full body when present; description is the summary.
        body = item.findtext(f"{{{_CONTENT_NS}}}encoded") or item.findtext("description")
        text = _strip_html(body)
        if not text:
            continue
        basis = guid or link
        if not basis:
            continue
        docs.append(
            Document(
                id=_doc_id(basis),
                title=title or "Untitled",
                url=link or guid,
                text=text,
            )
        )
    return docs


def _parse_atom(root: ET.Element) -> list[Document]:
    docs: list[Document] = []
    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        title = _strip_html(entry.findtext(f"{{{_ATOM_NS}}}title"))
        link = _atom_link(entry)
        entry_id = (entry.findtext(f"{{{_ATOM_NS}}}id") or "").strip()
        # content is the full body; summary is the abstract-length fallback.
        body = entry.findtext(f"{{{_ATOM_NS}}}content") or entry.findtext(f"{{{_ATOM_NS}}}summary")
        text = _strip_html(body)
        if not text:
            continue
        basis = entry_id or link
        if not basis:
            continue
        docs.append(
            Document(
                id=_doc_id(basis),
                title=title or "Untitled",
                url=link or entry_id,
                text=text,
            )
        )
    return docs


def parse_feed(xml_bytes: bytes) -> list[Document]:
    """Parse RSS 2.0 or Atom 1.0 bytes into Documents, auto-detecting the format.

    Entries with no body text (or no identifying guid/link) are skipped. An
    unrecognized root element raises ``ValueError`` rather than silently
    returning nothing.
    """
    root = ET.fromstring(xml_bytes)
    tag = _local(root.tag).lower()
    if tag == "rss":
        return _parse_rss(root)
    if tag == "feed":
        return _parse_atom(root)
    if tag == "rdf":  # RSS 1.0 shares the item element shape used by _parse_rss
        return _parse_rss(root)
    raise ValueError(f"unrecognized feed root element: <{_local(root.tag)}>")


def fetch_corpus(query: str, max_docs: int = 100) -> list[Document]:
    """Fetch a feed and return up to ``max_docs`` of its entries as Documents.

    For this adapter ``query`` is the feed URL (there is no server-side search on
    a plain feed), keeping the fetch signature uniform with the other sources.
    """
    docs = parse_feed(_get(query))
    time.sleep(REQUEST_DELAY_SECONDS)  # be a polite client before any follow-up
    return docs[:max_docs]
