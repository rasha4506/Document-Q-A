"""Tiny JSON registry of uploaded documents (Chroma is awkward to list documents from)."""
import json
import threading
from pathlib import Path

from app.models import DocumentInfo

_lock = threading.Lock()


class Registry:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict]:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2))

    def all(self) -> list[DocumentInfo]:
        return [DocumentInfo(**d) for d in self._load().values()]

    def get(self, doc_id: str) -> DocumentInfo | None:
        d = self._load().get(doc_id)
        return DocumentInfo(**d) if d else None

    def put(self, info: DocumentInfo) -> None:
        with _lock:
            data = self._load()
            data[info.id] = info.model_dump(mode="json")
            self._save(data)

    def remove(self, doc_id: str) -> None:
        with _lock:
            data = self._load()
            data.pop(doc_id, None)
            self._save(data)
