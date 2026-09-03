"""Centralized application configuration.

Every setting has a working default so the application can be cloned and run
with zero configuration (using the offline embedding/LLM tiers and the
bundled sample dataset). Production deployments override these via
environment variables or a `.env` file -- see `.env.example` for the full,
documented list.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderMode(str, Enum):
    """How a pluggable provider (embeddings or LLM) should be selected.

    AUTO tries the highest-quality option first and gracefully degrades:
    openai -> local (HuggingFace) -> offline (pure-Python, no downloads).
    The other values pin a single tier and raise loudly if it can't be used,
    which is what you want in a production deployment.
    """

    AUTO = "auto"
    OPENAI = "openai"
    LOCAL = "local"
    OFFLINE = "offline"


class DataSourceMode(str, Enum):
    """Where the ingestion pipeline pulls raw support documents from."""

    SAMPLE = "sample"
    HUGGINGFACE = "hf"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Database ---------------------------------------------------------
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag_support"

    # --- Provider selection -------------------------------------------------
    embedding_provider: ProviderMode = ProviderMode.AUTO
    llm_provider: ProviderMode = ProviderMode.AUTO

    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536

    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_embedding_dim: int = 384
    local_llm_model: str = "google/flan-t5-base"

    # Pure-Python offline fallback (HashingVectorizer). Fixed dimensionality
    # so the pgvector column type never has to change with the active tier.
    offline_embedding_dim: int = 512

    # The pgvector column is created with a fixed width (Postgres/pgvector
    # requires this). It must match whatever embedding_provider is actually
    # active in your deployment: 1536 for openai/text-embedding-3-small,
    # 384 for the local MiniLM model, 512 for the offline hashing tier
    # (the zero-config default). Changing providers to one with a different
    # width requires an Alembic migration + re-embedding the corpus -- see
    # docs/architecture.md for the trade-off this makes.
    vector_dim: int = 512

    # --- Data source ----------------------------------------------------
    data_source: DataSourceMode = DataSourceMode.SAMPLE

    # --- Chunking / retrieval ---------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    # --- API ----------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached after first call)."""
    return Settings()
