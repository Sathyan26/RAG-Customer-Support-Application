from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_support.api.schemas import DocumentSummary
from rag_support.storage.db import get_db
from rag_support.storage.models import Document, DocumentChunk

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentSummary])
def list_documents(
    session: Session = Depends(get_db),
    category: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, le=500),
) -> list[DocumentSummary]:
    stmt = (
        select(Document, func.count(DocumentChunk.id))
        .outerjoin(DocumentChunk, DocumentChunk.document_id == Document.id)
        .group_by(Document.id)
        .order_by(Document.id)
        .limit(limit)
    )
    if category:
        stmt = stmt.where(Document.category == category)
    if status:
        stmt = stmt.where(Document.status == status)

    return [
        DocumentSummary(
            id=doc.id,
            source=doc.source,
            category=doc.category,
            intent=doc.intent,
            title=doc.title,
            status=doc.status,
            chunk_count=chunk_count,
        )
        for doc, chunk_count in session.execute(stmt).all()
    ]
