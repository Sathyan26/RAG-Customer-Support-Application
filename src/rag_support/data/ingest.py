"""Ingestion stage: drain a `DataSource` into the `documents` table.

This is the first stage of the single pipeline (ingest -> clean -> embed)
that the CLI and API both call into -- see `rag_support/rag/pipeline.py` and
`rag_support/cli.py`. Nothing here talks to a specific source directly; it
only knows the `DataSource` interface, so `run_ingestion` works identically
whether it's fed the bundled sample or the live Hugging Face connector.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from rag_support.data.sources.base import DataSource
from rag_support.logging_config import get_logger
from rag_support.storage.repository import NewDocument, bulk_insert_documents

logger = get_logger(__name__)


@dataclass(slots=True)
class IngestionStats:
    source: str
    records_read: int
    documents_written: int


def run_ingestion(session: Session, source: DataSource, batch_size: int = 200) -> IngestionStats:
    records_read = 0
    documents_written = 0
    batch: list[NewDocument] = []

    def flush() -> None:
        nonlocal documents_written
        if not batch:
            return
        ids = bulk_insert_documents(session, batch)
        documents_written += len(ids)
        batch.clear()

    for record in source.fetch():
        records_read += 1
        batch.append(
            NewDocument(
                source=source.name,
                external_id=record.external_id,
                category=record.category,
                intent=record.intent,
                title=record.title,
                raw_text=record.text,
                doc_metadata=record.metadata,
            )
        )
        if len(batch) >= batch_size:
            flush()

    flush()
    logger.info(
        "Ingestion complete: source=%s records_read=%d documents_written=%d",
        source.name,
        records_read,
        documents_written,
    )
    return IngestionStats(
        source=source.name, records_read=records_read, documents_written=documents_written
    )
