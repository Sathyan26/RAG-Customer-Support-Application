"""Data-access layer: every read/write the rest of the app needs, in one place.

Keeping SQL/ORM queries out of the pipeline, API routers, and CLI keeps those
layers testable with plain mocks and means there is exactly one place that
knows how a "similar chunk" query is actually written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_support.storage.models import Conversation, Document, DocumentChunk, Message


@dataclass(slots=True)
class NewDocument:
    source: str
    raw_text: str
    external_id: str | None = None
    category: str | None = None
    intent: str | None = None
    title: str | None = None
    doc_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class NewChunk:
    chunk_index: int
    content: str
    token_count: int


def bulk_insert_documents(session: Session, docs: list[NewDocument]) -> list[int]:
    orm_docs = [
        Document(
            source=d.source,
            external_id=d.external_id,
            category=d.category,
            intent=d.intent,
            title=d.title,
            raw_text=d.raw_text,
            doc_metadata=d.doc_metadata,
            status="ingested",
        )
        for d in docs
    ]
    session.add_all(orm_docs)
    session.flush()
    return [d.id for d in orm_docs]


def get_documents_by_status(session: Session, status: str) -> list[Document]:
    stmt = select(Document).where(Document.status == status)
    return list(session.scalars(stmt))


def get_all_documents(session: Session, limit: int | None = None) -> list[Document]:
    stmt = select(Document).order_by(Document.id)
    if limit:
        stmt = stmt.limit(limit)
    return list(session.scalars(stmt))


def count_documents(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(Document)) or 0


def mark_document_cleaned(
    session: Session, document_id: int, clean_text: str, content_hash: str
) -> None:
    doc = session.get(Document, document_id)
    if doc is None:
        raise ValueError(f"Document {document_id} not found")
    doc.clean_text = clean_text
    doc.content_hash = content_hash
    doc.status = "cleaned"


def mark_document_rejected(session: Session, document_id: int, reason: str) -> None:
    doc = session.get(Document, document_id)
    if doc is None:
        raise ValueError(f"Document {document_id} not found")
    doc.status = f"rejected_{reason}"


def content_hash_exists(session: Session, content_hash: str) -> bool:
    """True if a *cleaned* document with this hash already exists.

    Checked against persisted state (not just the current run's in-memory
    set), so re-running ingestion + cleaning on top of an existing corpus
    still catches cross-run duplicates.
    """
    stmt = select(Document.id).where(
        Document.content_hash == content_hash, Document.status == "cleaned"
    )
    return session.scalars(stmt).first() is not None


def replace_chunks(
    session: Session, document_id: int, chunks: list[NewChunk]
) -> list[DocumentChunk]:
    """Delete any existing chunks for a document and insert the new set.

    Re-chunking is idempotent this way -- running the cleaning stage twice
    never leaves orphaned or duplicated chunks behind.
    """
    doc = session.get(Document, document_id)
    if doc is None:
        raise ValueError(f"Document {document_id} not found")

    existing = session.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == document_id)
    )
    for chunk in existing:
        session.delete(chunk)
    session.flush()

    orm_chunks = [
        DocumentChunk(
            document_id=document_id,
            chunk_index=c.chunk_index,
            content=c.content,
            token_count=c.token_count,
        )
        for c in chunks
    ]
    session.add_all(orm_chunks)
    session.flush()
    return orm_chunks


def get_chunks_missing_embeddings(session: Session, limit: int = 256) -> list[DocumentChunk]:
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.embedding.is_(None))
        .order_by(DocumentChunk.id)
        .limit(limit)
    )
    return list(session.scalars(stmt))


def set_chunk_embedding(
    session: Session, chunk_id: int, embedding: list[float], provider: str
) -> None:
    chunk = session.get(DocumentChunk, chunk_id)
    if chunk is None:
        raise ValueError(f"Chunk {chunk_id} not found")
    chunk.embedding = embedding
    chunk.embedding_provider = provider


@dataclass(slots=True)
class VectorSearchHit:
    chunk: DocumentChunk
    distance: float
    document_category: str | None
    document_title: str | None
    document_external_id: str | None


def vector_search(
    session: Session,
    query_embedding: list[float],
    top_k: int = 4,
    category: str | None = None,
) -> list[VectorSearchHit]:
    """Nearest-neighbour search over chunk embeddings using cosine distance.

    Results are sorted by ascending distance (most similar first).
    ``distance`` is pgvector's cosine distance (0 = identical direction, up
    to 2 = opposite); the RAG pipeline turns it into a similarity score for
    display. Always joins Document so category/title/source are available
    without a lazy-load per hit (avoids an N+1 query for a handful of rows).
    """
    stmt = (
        select(
            DocumentChunk,
            DocumentChunk.embedding.cosine_distance(query_embedding).label("distance"),
            Document.category,
            Document.title,
            Document.external_id,
        )
        .join(Document, DocumentChunk.document_id == Document.id)
        .where(DocumentChunk.embedding.is_not(None))
    )

    if category:
        stmt = stmt.where(Document.category == category)

    stmt = stmt.order_by("distance").limit(top_k)
    return [
        VectorSearchHit(
            chunk=row[0],
            distance=row[1],
            document_category=row[2],
            document_title=row[3],
            document_external_id=row[4],
        )
        for row in session.execute(stmt).all()
    ]


def create_conversation(session: Session) -> Conversation:
    conversation = Conversation()
    session.add(conversation)
    session.flush()
    return conversation


def get_conversation(session: Session, conversation_id: int) -> Conversation | None:
    return session.get(Conversation, conversation_id)


def add_message(
    session: Session,
    conversation_id: int,
    role: str,
    content: str,
    retrieved_chunk_ids: list[int] | None = None,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        retrieved_chunk_ids=retrieved_chunk_ids or [],
    )
    session.add(message)
    session.flush()
    return message


def get_conversation_history(session: Session, conversation_id: int) -> list[Message]:
    stmt = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
    return list(session.scalars(stmt))
