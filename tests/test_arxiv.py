"""arXiv Atom parsing — exercised against a fixture so no network is needed."""

from docqa.ingest.arxiv import parse_atom

# A trimmed but structurally faithful arXiv Atom response: three entries, the
# last with no abstract. Titles/summaries carry the newlines and double spaces
# arXiv really emits, so the whitespace normalization is under test too.
FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.01234v2</id>
    <title>Reciprocal Rank Fusion
  Outperforms Naive Score Blending</title>
    <summary>  We show that RRF is robust
across corpora and needs no tuning.  </summary>
    <author><name>A. Researcher</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/cs/0301001v1</id>
    <title>An Older Paper</title>
    <summary>Legacy identifier scheme abstract.</summary>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2306.99999v1</id>
    <title>Withdrawn, No Abstract</title>
    <summary>   </summary>
  </entry>
</feed>
"""


def test_parse_skips_entries_without_an_abstract():
    docs = parse_atom(FIXTURE)
    assert len(docs) == 2  # the third entry has an empty summary


def test_parse_extracts_and_normalizes_the_first_entry():
    doc = parse_atom(FIXTURE)[0]
    assert doc.id == "arxiv-2301.01234"  # version suffix stripped
    assert doc.title == "Reciprocal Rank Fusion Outperforms Naive Score Blending"
    assert doc.text == "We show that RRF is robust across corpora and needs no tuning."
    assert doc.url == "https://arxiv.org/abs/2301.01234"


def test_parse_preserves_old_style_identifiers():
    doc = parse_atom(FIXTURE)[1]
    assert doc.id == "arxiv-cs/0301001"
    assert doc.url == "https://arxiv.org/abs/cs/0301001"


def test_parse_empty_feed_returns_no_docs():
    empty = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert parse_atom(empty) == []
