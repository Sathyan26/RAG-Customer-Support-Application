"""FastAPI application factory + ASGI entrypoint (`uvicorn rag_support.api.main:app`)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_support.api.routers import chat, documents, health, ingest
from rag_support.config import get_settings
from rag_support.embeddings.factory import get_embedding_provider
from rag_support.llm.factory import get_llm_provider
from rag_support.logging_config import configure_logging, get_logger
from rag_support.rag.pipeline import RAGPipeline
from rag_support.storage.db import ensure_pgvector_extension

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    ensure_pgvector_extension()

    embedder = get_embedding_provider(settings)
    llm = get_llm_provider(settings)

    app.state.settings = settings
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.pipeline = RAGPipeline(embedder, llm, top_k=settings.top_k)

    logger.info(
        "rag-support API started (embedding_provider=%s, llm_provider=%s, data_source=%s)",
        embedder.name,
        llm.name,
        settings.data_source.value,
    )
    yield
    logger.info("rag-support API shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Support API",
        description=(
            "A retrieval-augmented customer support assistant. Ingestion, "
            "cleaning, embedding, and generation all live behind this one API."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Permissive CORS: this is a portfolio/demo service, not a multi-tenant
    # production API with a fixed set of trusted origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(ingest.router)

    return app


app = create_app()
