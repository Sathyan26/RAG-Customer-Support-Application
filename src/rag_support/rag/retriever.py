"""Retrieval: embed a query and find the most relevant chunks in Postgres."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from rag_support.embeddings.base import EmbeddingProvider
from rag_support.storage.repository import vector_search


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: int
    document_id: int
    content: str
    similarity: float
    category: str | None
    title: str | None
    source_id: str | None


class Retriever:
    def __init__(self, embedder: EmbeddingProvider, default_top_k: int = 4) -> None:
        self._embedder = embedder
        self._default_top_k = default_top_k

    def retrieve(
        self,
        session: Session,
        query: str,
        top_k: int | None = None,
        category: str | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self._embedder.embed_one(query)
        hits = vector_search(
            session, query_embedding, top_k=top_k or self._default_top_k, category=category
        )
        return [
            RetrievedChunk(
                chunk_id=hit.chunk.id,
                document_id=hit.chunk.document_id,
                content=hit.chunk.content,
                # pgvector's cosine_distance is 1 - cosine_similarity, so
                # similarity = 1 - distance (range [-1, 1]); clamp to [0, 1]
                # for a clean 0-100% display -- negative similarity isn't
                # meaningfully different from "unrelated" for this UI.
                similarity=round(max(0.0, min(1.0, 1 - hit.distance)), 4),
                category=hit.document_category,
                title=hit.document_title,
                source_id=hit.document_external_id,
            )
            for hit in hits
        ]
