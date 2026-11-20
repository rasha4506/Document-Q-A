import pytest

from app.chunking import chunk_pages, get_tokenizer
from app.parsing import Page

LONG = " ".join(f"word{i}" for i in range(400))


def ntok(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False).ids)


def test_chunks_respect_size_and_page():
    pages = [Page(1, LONG), Page(2, "short page text")]
    chunks = chunk_pages(pages, "d1", "f.pdf", chunk_size=100, overlap=20)
    assert all(ntok(c.text) <= 105 for c in chunks)  # re-tokenizing a slice may differ by a few tokens
    assert {c.page for c in chunks} == {1, 2}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert chunks[-1].text == "short page text"


def test_overlap_shared_text():
    chunks = [c for c in chunk_pages([Page(1, LONG)], "d", "f", 100, 30)]
    assert len(chunks) > 3
    a, b = chunks[0].text.split(), chunks[1].text.split()
    assert set(a) & set(b)  # consecutive chunks share words


def test_no_overlap_means_fewer_chunks():
    a = chunk_pages([Page(1, LONG)], "d", "f", 100, 0)
    b = chunk_pages([Page(1, LONG)], "d", "f", 100, 50)
    assert len(b) > len(a)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "x")], "d", "f", 50, 50)


def test_empty_pages_skipped():
    assert chunk_pages([Page(1, "   ")], "d", "f", 50, 10) == []
