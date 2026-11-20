"""Thin wrapper over a Chroma collection."""
from dataclasses import dataclass

import chromadb

from app.chunking import Chunk


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    file: str
    page: int
    text: str
    score: float  # cosine similarity, higher is better


class VectorStore:
    def __init__(self, path: str, collection: str):
        self.client = chromadb.PersistentClient(path=path, settings=chromadb.Settings(anonymized_telemetry=False))
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self.col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "file": c.file, "page": c.page} for c in chunks],
        )

    def query(self, vector: list[float], k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        where = None
        if doc_ids:
            where = {"doc_id": doc_ids[0]} if len(doc_ids) == 1 else {"doc_id": {"$in": doc_ids}}
        res = self.col.query(query_embeddings=[vector], n_results=k, where=where)
        return [
            Hit(cid, m["doc_id"], m["file"], int(m["page"]), doc, 1.0 - dist)
            for cid, m, doc, dist in zip(res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0])
        ]

    def delete_doc(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self.col.count()
