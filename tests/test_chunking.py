import pytest

from docqa.chunking import chunk_document, chunk_text
from docqa.models import Document


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t ") == []


def test_short_text_is_a_single_chunk():
    text = words(50)
    assert chunk_text(text, chunk_size=200, overlap=40) == [text]


def test_exact_boundary_is_a_single_chunk():
    text = words(200)
    assert chunk_text(text, chunk_size=200, overlap=40) == [text]


def test_overlap_repeats_words_across_consecutive_chunks():
    chunks = chunk_text(words(360), chunk_size=200, overlap=40)
    assert len(chunks) == 2
    first, second = (c.split() for c in chunks)
    assert first[-40:] == second[:40]


def test_every_word_appears_in_some_chunk():
    n = 999
    chunks = chunk_text(words(n), chunk_size=200, overlap=40)
    covered = {w for chunk in chunks for w in chunk.split()}
    assert covered == set(words(n).split())
    assert chunks[-1].split()[-1] == f"w{n - 1}"


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=-1)


def test_chunk_document_prefixes_title_and_numbers_chunks():
    doc = Document(id="d1", title="A Title", url="", text=words(300))
    chunks = chunk_document(doc, chunk_size=100, overlap=20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.doc_id == "d1" for c in chunks)
    assert chunks[0].text.startswith("A Title")
