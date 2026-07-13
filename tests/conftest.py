import pytest

from docqa.embeddings import HashingEmbedder
from docqa.models import Document
from docqa.store.sqlite_store import SQLiteVectorStore


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def store(tmp_path):
    s = SQLiteVectorStore(str(tmp_path / "index.db"))
    yield s
    s.close()


DOCS = [
    Document(
        id="doc-hypertension",
        title="Hypertension basics",
        url="local://doc-hypertension",
        text=(
            "Hypertension means persistently elevated blood pressure. Stage 2 "
            "hypertension is a reading of 140/90 mmHg or higher, and treatment "
            "combines lifestyle changes with antihypertensive medication."
        ),
    ),
    Document(
        id="doc-kafka",
        title="Kafka event streaming",
        url="local://doc-kafka",
        text=(
            "Apache Kafka is a distributed event streaming platform. Producers "
            "append records to partitioned topics and consumer groups read them "
            "in order, enabling event-driven microservice architectures."
        ),
    ),
    Document(
        id="doc-espresso",
        title="Espresso extraction",
        url="local://doc-espresso",
        text=(
            "Espresso extraction pushes hot water through finely ground coffee "
            "at roughly nine bars of pressure. Grind size, dose, and shot time "
            "control whether the espresso tastes sour, balanced, or bitter."
        ),
    ),
]


@pytest.fixture
def docs():
    return list(DOCS)
