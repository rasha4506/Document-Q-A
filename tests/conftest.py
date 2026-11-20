import pymupdf
import pytest

PAGES = [
    "Leave Policy. Full-time employees receive 25 days of annual leave per year. Unused leave may be carried over up to 5 days.",
    "Remote Work. Employees may work remotely up to three days per week with manager approval. Equipment is provided by the company.",
    "Expenses. Travel must be booked through the company portal. Meals are reimbursed up to 40 dollars per day.",
    "Security. Passwords must be rotated every 90 days and multi-factor authentication is mandatory for all systems.",
]


def make_pdf(pages: list[str] = PAGES, header: str = "ACME Corp Handbook") -> bytes:
    doc = pymupdf.open()
    for i, text in enumerate(pages, 1):
        page = doc.new_page()
        page.insert_text((72, 40), header, fontsize=9)
        page.insert_textbox(pymupdf.Rect(72, 80, 520, 700), text, fontsize=11)
        page.insert_text((280, 800), f"Page {i}", fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def pdf_bytes() -> bytes:
    return make_pdf()
