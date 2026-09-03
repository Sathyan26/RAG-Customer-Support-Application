"""Single command-line entrypoint for the whole application.

`pip install -e .` registers this as the `rag-support` command (see
`[project.scripts]` in pyproject.toml); it's also runnable as
`python -m rag_support.cli`. Every pipeline stage that the API exposes over
HTTP is available here too, plus operational commands (`init-db`, `serve`)
-- this file is what makes "run the whole thing" a single command instead of
a README full of scripts to run in the right order.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag_support.config import get_settings
from rag_support.logging_config import configure_logging

app = typer.Typer(add_completion=False, help="RAG customer support assistant -- CLI")
console = Console()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _setup() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def init_db() -> None:
    """Create the pgvector extension and run all Alembic migrations."""
    from alembic import command
    from alembic.config import Config

    _setup()
    from rag_support.storage.db import ensure_pgvector_extension

    console.print("[bold]Ensuring pgvector extension exists...[/bold]")
    ensure_pgvector_extension()

    console.print("[bold]Running Alembic migrations (upgrade head)...[/bold]")
    alembic_cfg = Config(str(_PROJECT_ROOT / "alembic.ini"))
    command.upgrade(alembic_cfg, "head")
    console.print("[green]Database is up to date.[/green]")


@app.command()
def ingest() -> None:
    """Run only the ingestion stage: pull raw documents into the database."""
    from rag_support.data.ingest import run_ingestion
    from rag_support.data.sources import get_data_source
    from rag_support.storage.db import session_scope

    _setup()
    settings = get_settings()
    source = get_data_source(settings.data_source.value)
    with session_scope() as session:
        stats = run_ingestion(session, source)
    console.print(stats)


@app.command()
def clean() -> None:
    """Run only the cleaning stage: normalize, de-dupe, and chunk."""
    from rag_support.cleaning.pipeline import run_cleaning
    from rag_support.storage.db import session_scope

    _setup()
    settings = get_settings()
    with session_scope() as session:
        stats = run_cleaning(session, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    console.print(stats)


@app.command()
def embed() -> None:
    """Run only the embedding stage: fill in vectors for un-embedded chunks."""
    from rag_support.embeddings.factory import get_embedding_provider
    from rag_support.embeddings.pipeline import run_embedding
    from rag_support.storage.db import session_scope

    _setup()
    settings = get_settings()
    provider = get_embedding_provider(settings)
    with session_scope() as session:
        stats = run_embedding(session, provider)
    console.print(stats)


@app.command()
def pipeline() -> None:
    """Run ingest + clean + embed as one operation (the full data pipeline)."""
    from rag_support.data.sources import get_data_source
    from rag_support.embeddings.factory import get_embedding_provider
    from rag_support.rag.pipeline import run_ingest_clean_embed
    from rag_support.storage.db import session_scope

    _setup()
    settings = get_settings()
    source = get_data_source(settings.data_source.value)
    provider = get_embedding_provider(settings)

    with session_scope() as session:
        stats = run_ingest_clean_embed(
            session, source, provider, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap
        )

    table = Table(title="Pipeline results")
    table.add_column("Stage")
    table.add_column("Result")
    table.add_row("Ingestion", f"{stats.ingestion.documents_written} documents written")
    table.add_row(
        "Cleaning",
        f"{stats.cleaning.documents_cleaned} cleaned, "
        f"{stats.cleaning.duplicates_rejected} duplicates, "
        f"{stats.cleaning.low_quality_rejected} low-quality rejected, "
        f"{stats.cleaning.chunks_created} chunks",
    )
    table.add_row("Embedding", f"{stats.embedding.chunks_embedded} chunks ({stats.embedding.provider})")
    console.print(table)


@app.command()
def serve(
    host: str | None = typer.Option(None, help="Overrides API_HOST from settings."),
    port: int | None = typer.Option(None, help="Overrides API_PORT from settings."),
    reload: bool = typer.Option(False, help="Enable uvicorn autoreload for local development."),
) -> None:
    """Run the FastAPI app with uvicorn."""
    import uvicorn

    _setup()
    settings = get_settings()
    uvicorn.run(
        "rag_support.api.main:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


@app.command()
def chat(
    conversation_id: int | None = typer.Option(None, help="Continue an existing conversation."),
    category: str | None = typer.Option(None, help="Restrict retrieval to one category."),
) -> None:
    """Interactive REPL against the RAG pipeline, for quick manual testing."""
    from rag_support.embeddings.factory import get_embedding_provider
    from rag_support.llm.factory import get_llm_provider
    from rag_support.rag.pipeline import RAGPipeline
    from rag_support.storage.db import session_scope

    _setup()
    settings = get_settings()
    embedder = get_embedding_provider(settings)
    llm = get_llm_provider(settings)
    rag_pipeline = RAGPipeline(embedder, llm, top_k=settings.top_k)

    console.print(
        f"[bold]rag-support chat[/bold] (embedding={embedder.name}, llm={llm.name}). "
        "Type 'exit' to quit.\n"
    )
    convo_id = conversation_id
    while True:
        question = console.input("[bold cyan]you>[/bold cyan] ")
        if question.strip().lower() in {"exit", "quit"}:
            break
        with session_scope() as session:
            result = rag_pipeline.chat(
                session, question, conversation_id=convo_id, category=category
            )
        convo_id = result.conversation_id
        console.print(f"[bold magenta]assistant>[/bold magenta] {result.answer}\n")
        if result.sources:
            console.print("[dim]Sources:[/dim]")
            for s in result.sources:
                console.print(f"[dim]  [{s.rank}] {s.category} · {s.title} (similarity={s.similarity})[/dim]")
        console.print()


if __name__ == "__main__":
    app()
