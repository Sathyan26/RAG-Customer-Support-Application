"""End-to-end integration tests: ingest -> clean -> embed -> retrieve -> chat,
all against the real Postgres + pgvector test database, using the offline
embedder/LLM tiers (no network, no API keys -- see tests/conftest.py)."""

from __future__ import annotations

import pytest

from rag_support.cleaning.pipeline import run_cleaning
from rag_support.data.ingest import run_ingestion
from rag_support.data.sources import get_data_source
from rag_support.embeddings.offline_embedder import OfflineEmbedder
from rag_support.embeddings.pipeline import run_embedding
from rag_support.llm.offline_llm import ExtractiveLLM
from rag_support.rag.pipeline import RAGPipeline
from rag_support.rag.retriever import Retriever
from rag_support.storage.repository import count_documents, get_conversation_history


def _seed(session) -> OfflineEmbedder:
    source = get_data_source("sample")
    run_ingestion(session, source)
    run_cleaning(session)
    embedder = OfflineEmbedder(dim=512)
    run_embedding(session, embedder)
    session.flush()
    return embedder


def test_full_data_pipeline_ingests_cleans_and_embeds_the_sample_dataset(db_session):
    _seed(db_session)

    assert count_documents(db_session) == 80

    from rag_support.storage.repository import get_documents_by_status

    cleaned = get_documents_by_status(db_session, "cleaned")
    rejected = get_documents_by_status(db_session, "rejected_duplicate")
    assert len(cleaned) == 74
    assert len(rejected) == 6  # the 6 deliberately-injected duplicates

    for doc in cleaned:
        assert doc.chunks, f"document {doc.id} was cleaned but has no chunks"
        for chunk in doc.chunks:
            assert chunk.embedding is not None
            assert chunk.embedding_provider == "offline"


def test_retriever_ranks_the_matching_chunk_first(db_session):
    embedder = _seed(db_session)
    retriever = Retriever(embedder, default_top_k=3)

    # Phrased close to the actual KB entry -- the offline embedder is a
    # lexical (hashing) similarity tier, not semantic, so a paraphrase with
    # little word overlap isn't guaranteed to rank the right chunk first
    # (that's exactly why it's not the recommended production tier; see
    # embeddings/offline_embedder.py). A near-literal query is what it's
    # actually good at, and is what this test checks.
    results = retriever.retrieve(db_session, "I forgot my password, how do I reset it")

    assert results
    assert results[0].category == "ACCOUNT"
    assert "password" in results[0].content.lower()
    # Results should come back best-first.
    assert all(
        results[i].similarity >= results[i + 1].similarity for i in range(len(results) - 1)
    )


def test_retriever_category_filter_narrows_results(db_session):
    embedder = _seed(db_session)
    retriever = Retriever(embedder, default_top_k=5)

    results = retriever.retrieve(db_session, "help me please", category="SHIPPING")

    assert results
    assert all(r.category == "SHIPPING" for r in results)


def test_rag_pipeline_chat_persists_a_new_conversation(db_session):
    embedder = _seed(db_session)
    pipeline = RAGPipeline(embedder, ExtractiveLLM(), top_k=3)

    result = pipeline.chat(db_session, "How do I cancel my subscription?")

    assert result.conversation_id is not None
    assert result.sources
    assert result.sources[0].category == "BILLING"

    history = get_conversation_history(db_session, result.conversation_id)
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[0].content == "How do I cancel my subscription?"
    assert history[1].retrieved_chunk_ids == [s.chunk_id for s in result.sources]


def test_rag_pipeline_chat_continues_an_existing_conversation(db_session):
    embedder = _seed(db_session)
    pipeline = RAGPipeline(embedder, ExtractiveLLM(), top_k=3)

    first = pipeline.chat(db_session, "How do I reset my password?")
    second = pipeline.chat(
        db_session, "What about two-factor auth?", conversation_id=first.conversation_id
    )

    assert second.conversation_id == first.conversation_id
    history = get_conversation_history(db_session, first.conversation_id)
    assert len(history) == 4  # 2 user turns + 2 assistant turns


def test_rag_pipeline_chat_rejects_an_unknown_conversation_id(db_session):
    embedder = _seed(db_session)
    pipeline = RAGPipeline(embedder, ExtractiveLLM(), top_k=3)

    with pytest.raises(ValueError, match="not found"):
        pipeline.chat(db_session, "hello", conversation_id=999_999)
