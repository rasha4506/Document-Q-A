import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.embeddings import get_embedder
from app.ingest import file_id, ingest_pdf
from app.llm import LLMClient, OpenAICompatClient
from app.models import AskRequest, AskResponse, DocumentInfo, UploadResponse
from app.qa import answer_question
from app.registry import Registry
from app.store import VectorStore

log = logging.getLogger("uvicorn.error")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class State:
    store: VectorStore
    registry: Registry
    llm: LLMClient | None = None


state = State()


def init_state() -> None:
    s = get_settings()
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    state.store = VectorStore(str(s.chroma_dir), s.collection_name)
    state.registry = Registry(s.data_dir / "registry.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_state()
    yield


app = FastAPI(title="Document Q&A (RAG)", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",  # any local dev port (Vite may pick 5174+)
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_llm() -> LLMClient:
    if state.llm is None:
        try:
            state.llm = OpenAICompatClient()
        except RuntimeError as e:
            raise HTTPException(503, str(e))
    return state.llm


@app.get("/health")
def health():
    return {"status": "ok", "chunks": state.store.count()}


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 25 MB)")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "Only PDF files are supported")

    existing = state.registry.get(file_id(data))
    if existing:
        return UploadResponse(document=existing, duplicate=True)

    s = get_settings()
    try:
        info = ingest_pdf(data, file.filename or "document.pdf", state.store, get_embedder(), s.chunk_size, s.chunk_overlap)
    except Exception as e:  # corrupt / encrypted PDFs
        raise HTTPException(422, f"Could not parse PDF: {e}")
    if info.chunks == 0:
        raise HTTPException(422, "No extractable text found (scanned PDFs need OCR, which is not supported)")
    (s.upload_dir / f"{info.id}.pdf").write_bytes(data)
    state.registry.put(info)
    return UploadResponse(document=info)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    s = get_settings()
    if not state.registry.all():
        raise HTTPException(409, "Upload a document first")
    llm = get_llm()
    try:
        return answer_question(
            req.question, state.store, get_embedder(), llm, req.top_k or s.top_k, s.min_score, req.doc_ids
        )
    except HTTPException:
        raise
    except Exception as e:  # LLM auth / rate limit / bad model name: surface it instead of a bare 500
        log.exception("ask failed")
        raise HTTPException(502, f"LLM request failed: {type(e).__name__}: {str(e)[:300]}")


@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    return state.registry.all()


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str):
    if not state.registry.get(doc_id):
        raise HTTPException(404, "Document not found")
    state.store.delete_doc(doc_id)
    state.registry.remove(doc_id)
    (get_settings().upload_dir / f"{doc_id}.pdf").unlink(missing_ok=True)
