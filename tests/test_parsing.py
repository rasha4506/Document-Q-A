from app.parsing import Page, clean_pages, parse_pdf
from tests.conftest import make_pdf


def test_headers_footers_removed(pdf_bytes):
    pages = parse_pdf(pdf_bytes)
    assert [p.page for p in pages] == [1, 2, 3, 4]
    for p in pages:
        assert "ACME" not in p.text
        assert "Page " not in p.text
    assert "25 days of annual leave" in pages[0].text


def test_hyphenation_and_whitespace():
    pages = [Page(i, f"Section {chr(64+i)} text   with inter-\nnational reach {chr(100+i)}\nand more {chr(110+i)}.") for i in range(1, 4)]
    out = clean_pages(pages)
    assert "international" in out[0].text
    assert "   " not in out[0].text


def test_short_docs_keep_repeated_lines():
    # With <3 pages we can't tell headers from content, so nothing is stripped.
    out = clean_pages([Page(1, "Hello"), Page(2, "Hello")])
    assert out[0].text == "Hello"
