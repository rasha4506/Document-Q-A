"""Token-based sliding-window chunking that never crosses a page boundary.

Keeping chunks inside one page means every chunk has exactly one page number to cite.
Token counts use the embedding model's own tokenizer, so chunk_size is what the model actually sees.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from app.config import get_settings
from app.parsing import Page


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    file: str
    page: int
    text: str


@lru_cache
def get_tokenizer() -> Tokenizer:
    from app.embeddings import get_embedder

    get_embedder()  # make sure the model (and its tokenizer.json) is downloaded
    path = next(Path(get_settings().model_cache).rglob("tokenizer.json"))
    tok = Tokenizer.from_file(str(path))
    tok.no_truncation()
    tok.no_padding()
    return tok


def chunk_pages(pages: list[Page], doc_id: str, file: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    tok = get_tokenizer()
    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for p in pages:
        if not p.text.strip():
            continue
        enc = tok.encode(p.text, add_special_tokens=False)
        offsets = enc.offsets
        start = 0
        while start < len(offsets):
            end = min(start + chunk_size, len(offsets))
            text = p.text[offsets[start][0] : offsets[end - 1][1]].strip()
            chunks.append(Chunk(f"{doc_id}:{p.page}:{len(chunks)}", doc_id, file, p.page, text))
            if end == len(offsets):
                break
            start += step
    return chunks
