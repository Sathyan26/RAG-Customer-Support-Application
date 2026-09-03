"""Embedding stage: fill in `document_chunks.embedding` for every chunk that
doesn't have one yet. Third stage of the pipeline (ingest -> clean ->
**embed**)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from rag_support.embeddings.base import EmbeddingProvider
from rag_support.logging_config import get_logger
from rag_support.storage.repository import get_chunks_missing_embeddings, set_chunk_embedding

logger = get_logger(__name__)


@dataclass(slots=True)
class EmbeddingStats:
    provider: str
    chunks_embedded: int


def run_embedding(
    session: Session, provider: EmbeddingProvider, batch_size: int = 64
) -> EmbeddingStats:
    total = 0
    while True:
        chunks = get_chunks_missing_embeddings(session, limit=batch_size)
        if not chunks:
            break
        vectors = provider.embed([c.content for c in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            set_chunk_embedding(session, chunk.id, vector, provider.name)
        session.flush()
        total += len(chunks)
        logger.info("Embedded %d chunks so far (provider=%s)", total, provider.name)

    return EmbeddingStats(provider=provider.name, chunks_embedded=total)
