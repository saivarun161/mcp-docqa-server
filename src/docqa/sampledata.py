"""Access to the small corpus + eval testset bundled inside the package.

The sample corpus is a dozen short, original summaries of common healthcare
topics. It exists so a fresh clone can ingest, search, and eval in under a
minute with no API key, no database, and no network access.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

_DATA = files("docqa").joinpath("data")


@contextmanager
def sample_corpus_path() -> Iterator[Path]:
    with as_file(_DATA.joinpath("sample_corpus.jsonl")) as path:
        yield path


@contextmanager
def sample_testset_path() -> Iterator[Path]:
    with as_file(_DATA.joinpath("sample_testset.jsonl")) as path:
        yield path
