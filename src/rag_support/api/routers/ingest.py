"""Trigger the ingest -> clean -> embed pipeline over HTTP.

Runs synchronously and returns once the whole pipeline finishes. That's a
deliberate simplification for a project this size (see docs/architecture.md
"Future improvements" -- a background task queue is the obvious next step
for a corpus large enough that this blocks the request for more than a few
seconds, e.g. the full Hugging Face dataset rather than the sample).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from rag_support.api.deps import get_app_settings, get_embedder
from rag_support.api.schemas import IngestResponse
from rag_support.config import Settings
from rag_support.data.sources import get_data_source
from rag_support.embeddings.base import EmbeddingProvider
from rag_support.rag.pipeline import run_ingest_clean_embed
from rag_support.storage.db import get_db

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
def trigger_ingest(
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
    embedder: EmbeddingProvider = Depends(get_embedder),
) -> IngestResponse:
    source = get_data_source(settings.data_source.value)
    stats = run_ingest_clean_embed(
        session,
        source,
        embedder,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    session.commit()

    return IngestResponse(
        data_source=stats.ingestion.source,
        records_read=stats.ingestion.records_read,
        documents_written=stats.ingestion.documents_written,
        documents_cleaned=stats.cleaning.documents_cleaned,
        duplicates_rejected=stats.cleaning.duplicates_rejected,
        low_quality_rejected=stats.cleaning.low_quality_rejected,
        chunks_created=stats.cleaning.chunks_created,
        chunks_embedded=stats.embedding.chunks_embedded,
        embedding_provider=stats.embedding.provider,
    )
