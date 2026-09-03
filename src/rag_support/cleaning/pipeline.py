"""Cleaning stage: normalize, de-duplicate, and chunk ingested documents.

Second stage of the pipeline (ingest -> **clean** -> embed). Operates on
every `documents` row with `status == "ingested"`, so it's safe to call
repeatedly as new documents arrive without re-processing what's already
been cleaned.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from rag_support.cleaning.chunker import chunk_text
from rag_support.cleaning.normalize import clean_text, content_hash, is_low_quality
from rag_support.config import get_settings
from rag_support.logging_config import get_logger
from rag_support.storage.repository import (
    NewChunk,
    content_hash_exists,
    get_documents_by_status,
    mark_document_cleaned,
    mark_document_rejected,
    replace_chunks,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class CleaningStats:
    documents_seen: int = 0
    documents_cleaned: int = 0
    duplicates_rejected: int = 0
    low_quality_rejected: int = 0
    chunks_created: int = 0


def run_cleaning(
    session: Session, chunk_size: int | None = None, overlap: int | None = None
) -> CleaningStats:
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    stats = CleaningStats()
    documents = get_documents_by_status(session, "ingested")

    for document in documents:
        stats.documents_seen += 1
        cleaned = clean_text(document.raw_text)

        if is_low_quality(cleaned):
            mark_document_rejected(session, document.id, "low_quality")
            stats.low_quality_rejected += 1
            continue

        doc_hash = content_hash(cleaned)
        if content_hash_exists(session, doc_hash):
            mark_document_rejected(session, document.id, "duplicate")
            stats.duplicates_rejected += 1
            continue

        mark_document_cleaned(session, document.id, cleaned, doc_hash)

        pieces = chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
        new_chunks = [
            NewChunk(chunk_index=p.index, content=p.text, token_count=p.word_count)
            for p in pieces
        ]
        replace_chunks(session, document.id, new_chunks)

        stats.documents_cleaned += 1
        stats.chunks_created += len(new_chunks)

    logger.info(
        "Cleaning complete: seen=%d cleaned=%d duplicates=%d low_quality=%d chunks=%d",
        stats.documents_seen,
        stats.documents_cleaned,
        stats.duplicates_rejected,
        stats.low_quality_rejected,
        stats.chunks_created,
    )
    return stats
