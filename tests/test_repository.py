"""Integration tests for the storage/repository layer against a real
Postgres + pgvector database (see tests/conftest.py)."""

from __future__ import annotations

from sqlalchemy import select

from rag_support.config import get_settings
from rag_support.storage.models import DocumentChunk
from rag_support.storage.repository import (
    NewChunk,
    NewDocument,
    bulk_insert_documents,
    content_hash_exists,
    get_documents_by_status,
    mark_document_cleaned,
    replace_chunks,
    set_chunk_embedding,
    vector_search,
)


def _unit_vector(index: int) -> list[float]:
    dim = get_settings().vector_dim
    vector = [0.0] * dim
    vector[index] = 1.0
    return vector


def test_bulk_insert_and_get_by_status(db_session):
    ids = bulk_insert_documents(
        db_session, [NewDocument(source="manual", raw_text="hello world")]
    )
    docs = get_documents_by_status(db_session, "ingested")
    assert len(docs) == 1
    assert docs[0].id == ids[0]


def test_replace_chunks_is_idempotent_not_additive(db_session):
    doc_id = bulk_insert_documents(db_session, [NewDocument(source="manual", raw_text="x")])[0]
    replace_chunks(db_session, doc_id, [NewChunk(chunk_index=0, content="a", token_count=1)])
    replace_chunks(
        db_session,
        doc_id,
        [
            NewChunk(chunk_index=0, content="b", token_count=1),
            NewChunk(chunk_index=1, content="c", token_count=1),
        ],
    )
    chunks = db_session.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
    ).all()
    assert {c.content for c in chunks} == {"b", "c"}


def test_content_hash_exists_only_matches_cleaned_documents(db_session):
    doc_id = bulk_insert_documents(db_session, [NewDocument(source="manual", raw_text="x")])[0]
    assert content_hash_exists(db_session, "some-hash") is False
    mark_document_cleaned(db_session, doc_id, "clean text", "some-hash")
    assert content_hash_exists(db_session, "some-hash") is True
    assert content_hash_exists(db_session, "a-different-hash") is False


def test_vector_search_orders_results_by_cosine_similarity(db_session):
    doc_id = bulk_insert_documents(
        db_session, [NewDocument(source="manual", raw_text="x", category="TEST")]
    )[0]
    chunks = replace_chunks(
        db_session,
        doc_id,
        [
            NewChunk(chunk_index=0, content="alpha", token_count=1),
            NewChunk(chunk_index=1, content="beta", token_count=1),
        ],
    )
    set_chunk_embedding(db_session, chunks[0].id, _unit_vector(0), "test")
    set_chunk_embedding(db_session, chunks[1].id, _unit_vector(1), "test")
    db_session.flush()

    hits = vector_search(db_session, _unit_vector(0), top_k=2)

    assert hits[0].chunk.id == chunks[0].id
    assert hits[0].distance < hits[1].distance
    assert hits[0].document_category == "TEST"


def test_vector_search_respects_category_filter(db_session):
    billing_id, tech_id = bulk_insert_documents(
        db_session,
        [
            NewDocument(source="manual", raw_text="x", category="BILLING"),
            NewDocument(source="manual", raw_text="y", category="TECHNICAL"),
        ],
    )
    billing_chunk = replace_chunks(
        db_session, billing_id, [NewChunk(chunk_index=0, content="billing", token_count=1)]
    )[0]
    tech_chunk = replace_chunks(
        db_session, tech_id, [NewChunk(chunk_index=0, content="tech", token_count=1)]
    )[0]
    set_chunk_embedding(db_session, billing_chunk.id, _unit_vector(0), "test")
    set_chunk_embedding(db_session, tech_chunk.id, _unit_vector(0), "test")
    db_session.flush()

    hits = vector_search(db_session, _unit_vector(0), top_k=10, category="TECHNICAL")

    assert len(hits) == 1
    assert hits[0].chunk.id == tech_chunk.id
