from datetime import datetime

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    id: str
    filename: str
    pages: int
    chunks: int
    uploaded_at: datetime


class UploadResponse(BaseModel):
    document: DocumentInfo
    duplicate: bool = False


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    doc_ids: list[str] | None = None


class Source(BaseModel):
    file: str
    page: int
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    found: bool
    sources: list[Source]


class LLMAnswer(BaseModel):
    """Structured output we require from the LLM."""

    answer: str
    found: bool
    cited_sources: list[int] = []
