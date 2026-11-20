"""LLM access through any OpenAI-compatible chat API (Groq by default), plus a fake for tests."""
import json
from typing import Protocol

from openai import OpenAI

from app.config import get_settings
from app.models import LLMAnswer

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.
If the answer is not in the context, set "found" to false and answer "I don't know based on the documents."
Never use outside knowledge. Be concise.
Respond with a JSON object: {"answer": string, "found": boolean, "cited_sources": [integers]}
where cited_sources lists the numbers of the passages you actually used."""


class LLMClient(Protocol):
    def answer(self, question: str, passages: list[str]) -> LLMAnswer: ...


def build_user_prompt(question: str, passages: list[str]) -> str:
    ctx = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    return f"Context:\n{ctx}\n\nQuestion: {question}"


class OpenAICompatClient:
    def __init__(self):
        s = get_settings()
        if not s.llm_api_key:
            raise RuntimeError("LLM_API_KEY is not set (see .env.example)")
        self.client = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
        self.model = s.llm_model

    def answer(self, question: str, passages: list[str]) -> LLMAnswer:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, passages)},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            return LLMAnswer(**json.loads(raw))
        except (json.JSONDecodeError, ValueError):
            return LLMAnswer(answer=raw, found=True, cited_sources=[])


class FakeLLM:
    """Deterministic stand-in: echoes the first passage."""

    def answer(self, question: str, passages: list[str]) -> LLMAnswer:
        return LLMAnswer(answer=passages[0][:200], found=True, cited_sources=[1])
