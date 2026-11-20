from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    data_dir: Path = ROOT / "data"
    embed_model: str = "BAAI/bge-small-en-v1.5"

    chunk_size: int = 100  # tokens (embedding model tokenizer); best in eval/results.md
    chunk_overlap: int = 20
    top_k: int = 5
    min_score: float = 0.65  # cosine floor; answerable questions scored >= 0.74, off-topic ~0.5

    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def model_cache(self) -> Path:
        return self.data_dir / "models"

    @property
    def collection_name(self) -> str:
        return f"chunks_cs{self.chunk_size}_ov{self.chunk_overlap}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
