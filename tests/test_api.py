import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.llm import FakeLLM


@pytest.fixture
def client(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path, chunk_size=100, chunk_overlap=20, min_score=0.3)
    s.model_cache  # noqa
    monkeypatch.setattr(main, "get_settings", lambda: s)
    # reuse the already-downloaded model
    from app import embeddings
    real = embeddings.get_embedder()
    monkeypatch.setattr(main, "get_embedder", lambda: real)
    with TestClient(main.app) as c:
        main.state.llm = FakeLLM()
        yield c
    main.state.llm = None


def test_full_flow(client, pdf_bytes):
    assert client.get("/documents").json() == []
    assert client.post("/ask", json={"question": "hi"}).status_code == 409

    r = client.post("/upload", files={"file": ("handbook.pdf", pdf_bytes, "application/pdf")})
    assert r.status_code == 200 and r.json()["duplicate"] is False
    doc_id = r.json()["document"]["id"]

    dup = client.post("/upload", files={"file": ("copy.pdf", pdf_bytes, "application/pdf")}).json()
    assert dup["duplicate"] is True and dup["document"]["id"] == doc_id
    assert len(client.get("/documents").json()) == 1

    a = client.post("/ask", json={"question": "How many days of annual leave do I get?"}).json()
    assert a["found"] and a["sources"][0]["page"] == 1

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get("/documents").json() == []
    assert client.delete(f"/documents/{doc_id}").status_code == 404


def test_rejects_non_pdf(client):
    r = client.post("/upload", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 400
