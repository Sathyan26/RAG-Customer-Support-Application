"""The RAG pipeline: the single orchestrator both the API and the CLI call.

`RAGPipeline.chat()` is the whole retrieve-then-generate flow for one turn
of conversation: retrieve relevant chunks, build grounded context, call the
configured LLM, and persist both the turn and which chunks it was grounded
in. `run_ingest_clean_embed()` is the equivalent for the *data* side --
ingestion, cleaning, and embedding run as one call instead of three
disconnected scripts, which is the whole point of this project's
architecture (see the module docstring in `rag_support/data/ingest.py`,
`cleaning/pipeline.py`, and `embeddings/pipeline.py` for what each stage
does on its own).
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from rag_support.cleaning.pipeline import CleaningStats, run_cleaning
from rag_support.data.ingest import IngestionStats, run_ingestion
from rag_support.data.sources.base import DataSource
from rag_support.embeddings.base import EmbeddingProvider
from rag_support.embeddings.pipeline import EmbeddingStats, run_embedding
from rag_support.llm.base import HistoryTurn, LLMProvider
from rag_support.logging_config import get_logger
from rag_support.rag.retriever import RetrievedChunk, Retriever
from rag_support.storage.repository import (
    add_message,
    create_conversation,
    get_conversation,
    get_conversation_history,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class SourceCitation:
    rank: int
    chunk_id: int
    document_id: int
    source_id: str | None
    category: str | None
    title: str | None
    similarity: float
    excerpt: str


@dataclass(slots=True)
class ChatResult:
    conversation_id: int
    answer: str
    sources: list[SourceCitation]


class RAGPipeline:
    def __init__(self, embedder: EmbeddingProvider, llm: LLMProvider, top_k: int = 4) -> None:
        self.embedder = embedder
        self.llm = llm
        self.retriever = Retriever(embedder, default_top_k=top_k)

    def chat(
        self,
        session: Session,
        question: str,
        conversation_id: int | None = None,
        category: str | None = None,
        top_k: int | None = None,
    ) -> ChatResult:
        if conversation_id is None:
            conversation_id = create_conversation(session).id
        elif get_conversation(session, conversation_id) is None:
            raise ValueError(f"Conversation {conversation_id} not found")

        history = [
            HistoryTurn(role=m.role, content=m.content)
            for m in get_conversation_history(session, conversation_id)
        ]

        retrieved = self.retriever.retrieve(session, question, top_k=top_k, category=category)
        context = [self._format_for_context(c) for c in retrieved]

        answer = self.llm.generate(question, context, history=history)

        add_message(session, conversation_id, role="user", content=question)
        add_message(
            session,
            conversation_id,
            role="assistant",
            content=answer,
            retrieved_chunk_ids=[c.chunk_id for c in retrieved],
        )

        sources = [
            SourceCitation(
                rank=i + 1,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                source_id=c.source_id,
                category=c.category,
                title=c.title,
                similarity=c.similarity,
                excerpt=c.content if len(c.content) <= 280 else c.content[:277] + "...",
            )
            for i, c in enumerate(retrieved)
        ]
        return ChatResult(conversation_id=conversation_id, answer=answer, sources=sources)

    @staticmethod
    def _format_for_context(chunk: RetrievedChunk) -> str:
        label = chunk.title or chunk.category or "Support KB"
        return f"({label}) {chunk.content}"


@dataclass(slots=True)
class FullPipelineStats:
    ingestion: IngestionStats
    cleaning: CleaningStats
    embedding: EmbeddingStats


def run_ingest_clean_embed(
    session: Session,
    source: DataSource,
    embedder: EmbeddingProvider,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> FullPipelineStats:
    """Run all three data-side stages back to back, as one logical operation."""
    ingestion_stats = run_ingestion(session, source)
    cleaning_stats = run_cleaning(session, chunk_size=chunk_size, overlap=overlap)
    embedding_stats = run_embedding(session, embedder)
    logger.info(
        "Full pipeline complete: %s | %s | %s", ingestion_stats, cleaning_stats, embedding_stats
    )
    return FullPipelineStats(
        ingestion=ingestion_stats, cleaning=cleaning_stats, embedding=embedding_stats
    )
