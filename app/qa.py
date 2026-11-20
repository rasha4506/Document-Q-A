"""Phase B: question -> embed -> retrieve -> prompt LLM -> answer + sources."""
from app.embeddings import Embedder
from app.llm import LLMClient
from app.models import AskResponse, Source
from app.store import Hit, VectorStore

NOT_FOUND = "I don't know based on the documents."


def retrieve(question: str, store: VectorStore, embedder: Embedder, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
    return store.query(embedder.embed_query(question), k, doc_ids)


def _snippet(text: str, n: int = 300) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def to_source(h: Hit) -> Source:
    return Source(file=h.file, page=h.page, snippet=_snippet(h.text), score=round(h.score, 3))


def answer_question(
    question: str,
    store: VectorStore,
    embedder: Embedder,
    llm: LLMClient,
    k: int,
    min_score: float,
    doc_ids: list[str] | None = None,
) -> AskResponse:
    hits = retrieve(question, store, embedder, k, doc_ids)
    if not hits or hits[0].score < min_score:
        return AskResponse(answer=NOT_FOUND, found=False, sources=[])

    out = llm.answer(question, [h.text for h in hits])
    if not out.found:
        return AskResponse(answer=out.answer or NOT_FOUND, found=False, sources=[])

    cited = [hits[i - 1] for i in dict.fromkeys(out.cited_sources) if 1 <= i <= len(hits)]
    return AskResponse(answer=out.answer, found=True, sources=[to_source(h) for h in (cited or hits)])
