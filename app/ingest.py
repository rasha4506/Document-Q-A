"""Phase A: PDF -> parse -> clean -> chunk -> embed -> store."""
import hashlib
from datetime import datetime, timezone

from app.chunking import chunk_pages
from app.embeddings import Embedder
from app.models import DocumentInfo
from app.parsing import parse_pdf
from app.store import VectorStore


def file_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def ingest_pdf(
    data: bytes, filename: str, store: VectorStore, embedder: Embedder, chunk_size: int, overlap: int
) -> DocumentInfo:
    doc_id = file_id(data)
    pages = parse_pdf(data)
    chunks = chunk_pages(pages, doc_id, filename, chunk_size, overlap)
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return DocumentInfo(
        id=doc_id, filename=filename, pages=len(pages), chunks=len(chunks), uploaded_at=datetime.now(timezone.utc)
    )
