from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_support.api.deps import get_embedder, get_llm
from rag_support.api.schemas import HealthResponse
from rag_support.embeddings.base import EmbeddingProvider
from rag_support.llm.base import LLMProvider
from rag_support.storage.db import get_db
from rag_support.storage.models import DocumentChunk
from rag_support.storage.repository import count_documents

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    session: Session = Depends(get_db),
    embedder: EmbeddingProvider = Depends(get_embedder),
    llm: LLMProvider = Depends(get_llm),
) -> HealthResponse:
    try:
        document_count = count_documents(session)
        chunk_count = session.scalar(select(func.count()).select_from(DocumentChunk)) or 0
        database_status = "ok"
    except Exception:  # pragma: no cover - exercised only when the DB is down
        database_status = "error"
        document_count = 0
        chunk_count = 0

    return HealthResponse(
        status="ok" if database_status == "ok" else "degraded",
        database=database_status,
        embedding_provider=embedder.name,
        llm_provider=llm.name,
        document_count=document_count,
        chunk_count=chunk_count,
    )
