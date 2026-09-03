"""FastAPI dependency wiring.

Providers (embedder, LLM, pipeline) are expensive to construct -- loading a
local model, opening an OpenAI client -- so they're built exactly once at
app startup (see `main.lifespan`) and stashed on `app.state`. These
dependency functions just read them back out per-request; `get_db` (from
`storage.db`) is the one dependency that's genuinely per-request, since each
request needs its own database session/transaction.
"""

from __future__ import annotations

from fastapi import Request

from rag_support.config import Settings, get_settings
from rag_support.embeddings.base import EmbeddingProvider
from rag_support.llm.base import LLMProvider
from rag_support.rag.pipeline import RAGPipeline


def get_app_settings() -> Settings:
    return get_settings()


def get_embedder(request: Request) -> EmbeddingProvider:
    return request.app.state.embedder


def get_llm(request: Request) -> LLMProvider:
    return request.app.state.llm


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline
