"""Generic RSS/Atom feed parsing — exercised against fixtures, so no network."""

import pytest

from docqa.ingest.feed import parse_feed

# A trimmed but structurally faithful RSS 2.0 feed. Bodies carry HTML and
# entities (so tag stripping and unescaping are under test), one item prefers
# content:encoded over description, one item has an empty body, and one has no
# guid but does have a link.
RSS_FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Example Blog</title>
    <item>
      <title>Shipping &amp; Retrieval</title>
      <link>https://example.com/posts/1</link>
      <guid>https://example.com/posts/1</guid>
      <description>A short &lt;b&gt;summary&lt;/b&gt; only.</description>
      <content:encoded><![CDATA[<p>The <em>full</em> body   with markup.</p>]]></content:encoded>
    </item>
    <item>
      <title>Summary Only</title>
      <link>https://example.com/posts/2</link>
      <description>&lt;div&gt;Just a description here.&lt;/div&gt;</description>
    </item>
    <item>
      <title>Empty Body</title>
      <link>https://example.com/posts/3</link>
      <guid>https://example.com/posts/3</guid>
      <description>   </description>
    </item>
  </channel>
</rss>
"""

# Atom 1.0: multiple <link>s per entry (the alternate is the readable page),
# content preferred over summary, and a last entry with no body to be skipped.
ATOM_FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Example Atom</title>
  <entry>
    <title>First Entry</title>
    <link rel="self" href="https://example.com/atom/self"/>
    <link rel="alternate" href="https://example.com/atom/1"/>
    <id>urn:uuid:0001</id>
    <summary>Short summary.</summary>
    <content type="html">&lt;p&gt;Full &lt;b&gt;content&lt;/b&gt; body.&lt;/p&gt;</content>
  </entry>
  <entry>
    <title>Summary Fallback</title>
    <link href="https://example.com/atom/2"/>
    <id>urn:uuid:0002</id>
    <summary>Only a summary is present.</summary>
  </entry>
  <entry>
    <title>No Body</title>
    <link href="https://example.com/atom/3"/>
    <id>urn:uuid:0003</id>
  </entry>
</feed>
"""


def test_rss_skips_entries_without_a_body():
    docs = parse_feed(RSS_FIXTURE)
    assert len(docs) == 2  # the empty-description item is dropped


def test_rss_prefers_content_encoded_and_strips_html():
    doc = parse_feed(RSS_FIXTURE)[0]
    assert doc.title == "Shipping & Retrieval"
    assert doc.text == "The full body with markup."  # content:encoded, tags gone, ws collapsed
    assert doc.url == "https://example.com/posts/1"


def test_rss_falls_back_to_description():
    doc = parse_feed(RSS_FIXTURE)[1]
    assert doc.text == "Just a description here."


def test_atom_skips_entries_without_a_body():
    docs = parse_feed(ATOM_FIXTURE)
    assert len(docs) == 2  # the no-body entry is dropped


def test_atom_prefers_content_and_picks_the_alternate_link():
    doc = parse_feed(ATOM_FIXTURE)[0]
    assert doc.title == "First Entry"
    assert doc.text == "Full content body."  # content wins over summary, tags stripped
    assert doc.url == "https://example.com/atom/1"  # alternate, not the self link


def test_atom_falls_back_to_summary():
    doc = parse_feed(ATOM_FIXTURE)[1]
    assert doc.text == "Only a summary is present."


def test_ids_are_stable_and_differ_per_entry():
    first = parse_feed(RSS_FIXTURE)
    second = parse_feed(RSS_FIXTURE)
    assert first[0].id == second[0].id  # deterministic across runs
    assert first[0].id != first[1].id  # distinct entries, distinct ids
    assert first[0].id.startswith("feed-")


def test_unrecognized_root_raises():
    with pytest.raises(ValueError, match="unrecognized feed root"):
        parse_feed(b"<html><body>not a feed</body></html>")


def test_empty_feeds_return_no_docs():
    assert parse_feed(b'<rss version="2.0"><channel></channel></rss>') == []
    assert parse_feed(b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>') == []
