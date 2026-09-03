"""Pydantic request/response models for the API -- kept separate from the
ORM models (`storage/models.py`) so the wire format can evolve independently
of the database schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    conversation_id: int | None = Field(
        default=None, description="Omit to start a new conversation."
    )
    category: str | None = Field(
        default=None, description="Restrict retrieval to one category, e.g. 'BILLING'."
    )
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceCitationResponse(BaseModel):
    rank: int
    chunk_id: int
    document_id: int
    source_id: str | None
    category: str | None
    title: str | None
    similarity: float
    excerpt: str


class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    sources: list[SourceCitationResponse]
    embedding_provider: str
    llm_provider: str


class IngestResponse(BaseModel):
    data_source: str
    records_read: int
    documents_written: int
    documents_cleaned: int
    duplicates_rejected: int
    low_quality_rejected: int
    chunks_created: int
    chunks_embedded: int
    embedding_provider: str


class DocumentSummary(BaseModel):
    id: int
    source: str
    category: str | None
    intent: str | None
    title: str | None
    status: str
    chunk_count: int


class HealthResponse(BaseModel):
    status: str
    database: str
    embedding_provider: str
    llm_provider: str
    document_count: int
    chunk_count: int
