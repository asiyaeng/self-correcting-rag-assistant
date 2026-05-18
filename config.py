"""
app/config.py
Central configuration loaded from environment / .env file.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "groq", "anthropic"] = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    groq_api_key: str = ""
    anthropic_api_key: str = ""

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_provider: Literal["openai", "local"] = "local"
    embedding_model_local: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_openai: str = "text-embedding-3-small"

    # ── Vector store ──────────────────────────────────────────────────────
    vector_store: Literal["chroma", "faiss"] = "chroma"
    chroma_persist_dir: str = "./chroma_db"
    faiss_index_path: str = "./faiss_index"

    # ── Retrieval ─────────────────────────────────────────────────────────
    top_k: int = Field(default=5, ge=1, le=20)
    relevance_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_retries: int = Field(default=2, ge=0, le=5)

    # ── Chunking ──────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Web search (bonus) ────────────────────────────────────────────────
    enable_web_search: bool = False
    tavily_api_key: str = ""

    # ── App ───────────────────────────────────────────────────────────────
    app_title: str = "RAG Documentation Assistant"
    log_level: str = "INFO"
    feedback_store_path: str = "./feedback.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
