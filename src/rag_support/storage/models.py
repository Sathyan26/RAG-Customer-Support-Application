"""SQLAlchemy ORM models for the storage layer.

Four tables carry the whole pipeline's state:

- ``documents``       raw + cleaned support content as ingested
- ``document_chunks`` retrieval-sized pieces of a document, each with an
                       embedding vector (pgvector column)
- ``conversations``   one row per chat session
- ``messages``        turn-by-turn history, tagged with which chunks were
                       retrieved to ground each assistant reply

Keeping ingestion, cleaning status, and embeddings on the same schema (rather
than separate ad-hoc files/pickles per pipeline stage) is what makes this a
single storage layer instead of a chain of scripts passing files around.
"""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from rag_support.config import get_settings


class Base(DeclarativeBase):
    pass


def _vector_dim() -> int:
    return get_settings().vector_dim


class Document(Base):
    """A single unit of support knowledge before it is split into chunks.

    In the bundled sample dataset and the Hugging Face Bitext source, a
    "document" is one support entry (an intent/category plus its canonical
    answer); nothing stops a future data source from ingesting long-form
    articles instead -- the chunker downstream handles both.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_id: Mapped[str | None] = mapped_column(String(128), index=True)
    source: Mapped[str] = mapped_column(String(64))  # "sample" | "hf" | "manual"
    category: Mapped[str | None] = mapped_column(String(128), index=True)
    intent: Mapped[str | None] = mapped_column(String(128), index=True)
    title: Mapped[str | None] = mapped_column(String(512))
    raw_text: Mapped[str] = mapped_column(Text)
    clean_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="ingested")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Document(id={self.id!r}, source={self.source!r}, category={self.category!r})"


class DocumentChunk(Base):
    """A retrieval-sized slice of a document, with its embedding."""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_vector_dim()), nullable=True)
    embedding_provider: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    document: Mapped["Document"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DocumentChunk(id={self.id!r}, document_id={self.document_id!r})"


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    retrieved_chunk_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
