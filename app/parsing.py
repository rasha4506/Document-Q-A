"""PDF -> cleaned text per page (page numbers are 1-based, as a human reads them)."""
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class Page:
    page: int
    text: str


def extract_pages(source: Path | bytes) -> list[Page]:
    doc = pymupdf.open(stream=source, filetype="pdf") if isinstance(source, bytes) else pymupdf.open(source)
    with doc:
        return [Page(i + 1, p.get_text("text", sort=True)) for i, p in enumerate(doc)]


_PAGE_NUM = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.IGNORECASE)


def _norm(line: str) -> str:
    # Digits vary between pages in headers like "Policy v2 - page 3"; ignore them when spotting repeats.
    return re.sub(r"\d+", "#", line.strip().lower())


def clean_pages(pages: list[Page], repeat_ratio: float = 0.5) -> list[Page]:
    """Drop headers/footers (lines repeated on most pages), bare page numbers, and tidy whitespace."""
    n = len(pages)
    counts: Counter[str] = Counter()
    for p in pages:
        counts.update({_norm(l) for l in p.text.splitlines() if l.strip()})
    repeated = {k for k, c in counts.items() if n >= 3 and c / n > repeat_ratio}

    out = []
    for p in pages:
        lines = [
            l.strip()
            for l in p.text.splitlines()
            if l.strip() and _norm(l) not in repeated and not _PAGE_NUM.match(l)
        ]
        text = "\n".join(lines)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
        text = re.sub(r"(?<![.:;!?])\n(?=[a-z])", " ", text)  # join lines broken mid-sentence
        text = re.sub(r"[ \t]+", " ", text).strip()
        out.append(Page(p.page, text))
    return out


def parse_pdf(source: Path | bytes) -> list[Page]:
    return clean_pages(extract_pages(source))
