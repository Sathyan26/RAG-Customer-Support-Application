"""Shared pytest fixtures.

Tests run against a real Postgres + pgvector database (a `rag_support_test`
database, migrated with the project's actual Alembic revisions) rather than
mocks or SQLite -- pgvector's vector type and the cosine-distance queries in
`storage/repository.py` aren't meaningfully testable any other way, and
running against the real migrations catches schema drift that a
`Base.metadata.create_all()` shortcut would miss. See docs/setup.md for how
to provision the test database locally; CI provisions it via a Postgres
service container (`.github/workflows/ci.yml`).

Provider env vars default to the offline tiers so the suite needs no API
keys and no model downloads -- see `embeddings/offline_embedder.py` and
`llm/offline_llm.py`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import text

os.environ.setdefault(
    "DATABASE_URL",
    os.environ.get(
        "TEST_DATABASE_URL", "postgresql+psycopg://rag:rag@localhost:5432/rag_support_test"
    ),
)
os.environ.setdefault("EMBEDDING_PROVIDER", "offline")
os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("DATA_SOURCE", "sample")

from rag_support.config import get_settings  # noqa: E402
from rag_support.storage.db import (  # noqa: E402
    ensure_pgvector_extension,
    get_engine,
    get_session_factory,
    reset_engine_for_tests,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def _migrated_database():
    get_settings.cache_clear()
    ensure_pgvector_extension()

    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config(str(_PROJECT_ROOT / "alembic.ini"))
    command.upgrade(alembic_cfg, "head")

    yield
    reset_engine_for_tests()


@pytest.fixture(autouse=True)
def _clean_tables():
    """Truncate every table after each test so tests don't leak state."""
    yield
    with get_engine().begin() as conn:
        conn.execute(
            text(
                "TRUNCATE messages, document_chunks, documents, conversations "
                "RESTART IDENTITY CASCADE"
            )
        )


@pytest.fixture
def db_session():
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    finally:
        session.close()
