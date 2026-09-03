"""Integration tests for the FastAPI app, exercised through TestClient
(which runs the real lifespan handler, so providers are wired up exactly
as they would be under uvicorn) against the real test database."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_support.api.main import app
from rag_support.cleaning.pipeline import run_cleaning
from rag_support.data.ingest import run_ingestion
from rag_support.data.sources import get_data_source
from rag_support.embeddings.offline_embedder import OfflineEmbedder
from rag_support.embeddings.pipeline import run_embedding


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _seed(db_session) -> None:
    run_ingestion(db_session, get_data_source("sample"))
    run_cleaning(db_session)
    run_embedding(db_session, OfflineEmbedder(dim=512))
    # Commit (not just flush): the API serves each request from its own
    # session via get_db, so seeded data must actually be committed to be
    # visible to it.
    db_session.commit()


def test_health_reports_providers_and_live_counts(client, db_session):
    _seed(db_session)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "ok",
        "database": "ok",
        "embedding_provider": "offline",
        "llm_provider": "offline",
        "document_count": 80,
        "chunk_count": 74,
    }


def test_health_works_against_an_empty_database(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["document_count"] == 0


def test_documents_endpoint_lists_and_filters_by_category(client, db_session):
    _seed(db_session)
    response = client.get("/documents", params={"category": "BILLING", "limit": 5})
    assert response.status_code == 200
    docs = response.json()
    assert docs
    assert all(d["category"] == "BILLING" for d in docs)


def test_chat_returns_a_grounded_answer_with_ranked_sources(client, db_session):
    _seed(db_session)
    response = client.post("/chat", json={"question": "How do I reset my password?"})
    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"]
    assert body["sources"]
    assert body["sources"][0]["category"] == "ACCOUNT"
    assert body["embedding_provider"] == "offline"
    assert body["llm_provider"] == "offline"


def test_chat_continues_an_existing_conversation(client, db_session):
    _seed(db_session)
    first = client.post("/chat", json={"question": "How do I reset my password?"}).json()
    payload = {
        "question": "What about two-factor auth?",
        "conversation_id": first["conversation_id"],
    }
    second = client.post("/chat", json=payload).json()
    assert second["conversation_id"] == first["conversation_id"]


def test_chat_returns_404_for_an_unknown_conversation_id(client, db_session):
    _seed(db_session)
    response = client.post("/chat", json={"question": "hi", "conversation_id": 999_999})
    assert response.status_code == 404


def test_chat_rejects_an_empty_question(client):
    response = client.post("/chat", json={"question": ""})
    assert response.status_code == 422


def test_ingest_endpoint_runs_the_full_pipeline(client):
    response = client.post("/ingest")
    assert response.status_code == 200
    body = response.json()
    assert body["records_read"] == 80
    assert body["documents_cleaned"] == 74
    assert body["chunks_embedded"] == 74
