import pytest

from app.embeddings import get_embedder
from app.ingest import ingest_pdf
from app.qa import answer_question, retrieve
from app.store import VectorStore
from app.llm import FakeLLM


@pytest.fixture
def store(tmp_path, pdf_bytes):
    st = VectorStore(str(tmp_path / "chroma"), "test")
    ingest_pdf(pdf_bytes, "handbook.pdf", st, get_embedder(), 100, 20)
    return st


@pytest.mark.parametrize(
    "question,page",
    [
        ("How many days of annual leave do I get?", 1),
        ("Can I work from home?", 2),
        ("What is the daily meal allowance?", 3),
        ("How often must passwords change?", 4),
    ],
)
def test_retrieval_finds_right_page(store, question, page):
    hits = retrieve(question, store, get_embedder(), 3)
    assert page in [h.page for h in hits]
    assert hits[0].page == page


def test_off_topic_is_not_found(store):
    r = answer_question("What is the capital of Mongolia?", store, get_embedder(), FakeLLM(), 3, min_score=0.6)
    assert r.found is False and r.sources == []


def test_answer_has_sources(store):
    r = answer_question("How many days of annual leave do I get?", store, get_embedder(), FakeLLM(), 3, 0.3)
    assert r.found and r.sources[0].file == "handbook.pdf" and r.sources[0].page == 1
