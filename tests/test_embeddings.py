import numpy as np
import pytest

from docqa.embeddings import HashingEmbedder, get_embedder


def test_shape_dtype_and_normalization():
    emb = HashingEmbedder()
    matrix = emb.embed(["hello world", "another text entirely"])
    assert matrix.shape == (2, emb.dim)
    assert matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)


def test_empty_batch_and_empty_text():
    emb = HashingEmbedder()
    assert emb.embed([]).shape == (0, emb.dim)
    # embedding of empty text is the zero vector (norm guard avoids NaN)
    assert np.allclose(emb.embed([""])[0], 0.0)


def test_deterministic_across_instances():
    a = HashingEmbedder().embed(["sepsis bundle lactate antibiotics"])
    b = HashingEmbedder().embed(["sepsis bundle lactate antibiotics"])
    assert np.array_equal(a, b)


def test_similar_texts_score_higher_than_unrelated():
    emb = HashingEmbedder()
    query, related, unrelated = emb.embed(
        [
            "symptoms of heart attack",
            "heart attack symptoms include chest pain",
            "quarterly revenue grew nine percent",
        ]
    )
    assert float(query @ related) > float(query @ unrelated)


def test_get_embedder_selects_backend(monkeypatch):
    monkeypatch.setenv("DOCQA_EMBEDDINGS", "hash")
    assert isinstance(get_embedder(), HashingEmbedder)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DOCQA_EMBEDDINGS", "auto")
    assert isinstance(get_embedder(), HashingEmbedder)  # auto without a key -> hash
    with pytest.raises(ValueError):
        get_embedder("nonsense")
