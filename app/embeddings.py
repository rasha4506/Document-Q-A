"""Local, free embeddings via fastembed (ONNX runtime, no PyTorch)."""
from functools import lru_cache

from fastembed import TextEmbedding

from app.config import get_settings


class Embedder:
    def __init__(self, model_name: str, cache_dir: str):
        self.model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.passage_embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        # query_embed adds bge's "Represent this sentence for searching..." instruction
        return next(iter(self.model.query_embed([text]))).tolist()


@lru_cache
def get_embedder() -> Embedder:
    s = get_settings()
    return Embedder(s.embed_model, str(s.model_cache))
