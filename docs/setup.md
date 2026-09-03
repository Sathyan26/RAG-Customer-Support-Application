# Setup

Two ways to run this: Docker Compose (fastest, fewest moving parts) or a
local Python environment against a Postgres instance you manage yourself
(what this project was actually built and verified against, step by step,
in an environment with no Docker daemon available — see
[architecture.md](architecture.md) for why that matters). Pick whichever
fits; both are documented in full.

## Option A: Docker Compose

**Prerequisites:** Docker and Docker Compose.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/rag-customer-support.git
cd rag-customer-support
cp .env.example .env    # optional -- defaults work with zero changes
docker compose up --build
```

What this starts:

| Service | Port | What it is |
|---|---|---|
| `db` | 5432 | Postgres 16 + pgvector (`pgvector/pgvector:pg16` image) |
| `api` | 8000 | The FastAPI app -- runs `rag-support init-db` then `rag-support serve` on container start |
| `frontend` | 8501 | The Streamlit chat UI, pointed at `api` by service name |

Nothing is pre-ingested — hit `POST http://localhost:8000/ingest` once (or
`docker compose exec api rag-support pipeline`) to load the bundled sample
knowledge base before chatting. To stop everything and wipe the database
volume: `docker compose down -v` (or `make docker-down`).

To use real OpenAI models instead of the offline tiers, set in `.env`
before `docker compose up`:

```dotenv
OPENAI_API_KEY=sk-...
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai
```

## Option B: Local Python + a Postgres you run yourself

**Prerequisites:** Python 3.11+, a Postgres 16+ server with the pgvector
extension installed (not necessarily enabled yet — the app enables it).

### 1. Install Postgres + pgvector

Commands below are for Debian/Ubuntu (what this project was built on);
adjust for your OS (Homebrew: `brew install postgresql@16 pgvector`; or
just run `docker compose up -d db` from this repo and skip straight to
step 3, using Docker *only* for the database).

```bash
sudo apt-get install -y postgresql-16 postgresql-16-pgvector
sudo service postgresql start
```

### 2. Create the role and database

```bash
sudo -u postgres psql -c "CREATE ROLE rag WITH LOGIN PASSWORD 'rag' SUPERUSER;"
sudo -u postgres psql -c "CREATE DATABASE rag_support OWNER rag;"
```

(`SUPERUSER` here is a local-dev convenience so the app can run
`CREATE EXTENSION IF NOT EXISTS vector` itself on first startup; a real
deployment would instead have an operator run that one statement once with
elevated privileges and grant the app role less.)

### 3. Install the Python package

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai,datasets,frontend]"
```

Extras, install only what you need:

| Extra | Adds | Needed for |
|---|---|---|
| `openai` | `openai` SDK | `EMBEDDING_PROVIDER=openai` / `LLM_PROVIDER=openai` |
| `local-ml` | `sentence-transformers`, `transformers`, `torch` | `EMBEDDING_PROVIDER=local` / `LLM_PROVIDER=local` (downloads model weights from the HF Hub on first use) |
| `datasets` | HuggingFace `datasets` | `DATA_SOURCE=hf` |
| `frontend` | `streamlit` | the chat UI |
| `dev` | `pytest`, `ruff`, etc. | running tests / linting |

### 4. Configure

```bash
cp .env.example .env
```

Every setting has a working default (see the table in `.env.example`); at
minimum for local dev, confirm `DATABASE_URL` matches what you created in
step 2:

```dotenv
DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5432/rag_support
```

### 5. Initialize the database and run the pipeline

```bash
rag-support init-db     # CREATE EXTENSION vector; alembic upgrade head
rag-support pipeline    # ingest -> clean -> embed the sample dataset
```

Expected output looks like:

```
                             Pipeline results
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Stage     ┃ Result                                                      ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Ingestion │ 80 documents written                                        │
│ Cleaning  │ 74 cleaned, 6 duplicates, 0 low-quality rejected, 74 chunks │
│ Embedding │ 74 chunks (offline)                                         │
└───────────┴─────────────────────────────────────────────────────────────┘
```

(This is the real output this project produced running the sample dataset
through the offline tier -- the 6 duplicates are deliberately injected by
`scripts/generate_sample_dataset.py`, see
[data_pipeline.md](data_pipeline.md).)

### 6. Run it

```bash
rag-support serve              # API on http://localhost:8000 (docs at /docs)
rag-support chat                # or: interactive terminal chat, no server needed
streamlit run frontend/streamlit_app.py   # chat UI, needs the API running
```

Individual pipeline stages are also available standalone, if you want to
run (or re-run) just one:

```bash
rag-support ingest    # DataSource -> documents table
rag-support clean     # normalize, de-dupe, chunk
rag-support embed     # fill in vectors for chunks that don't have one yet
```

## Running the tests

Tests need their own database (kept separate from your dev database so
`pytest` truncating tables between tests never touches data you're looking
at):

```bash
sudo -u postgres psql -c "CREATE DATABASE rag_support_test OWNER rag;"
sudo -u postgres psql -d rag_support_test -c "CREATE EXTENSION IF NOT EXISTS vector;"

export TEST_DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5432/rag_support_test
pytest -v          # or: make test
ruff check src tests   # or: make lint
```

The suite (50 tests) runs entirely against the offline embedding/LLM tiers,
so it needs no API keys. CI (`.github/workflows/ci.yml`) does the same
thing against a `pgvector/pgvector:pg16` service container.

## Evaluating retrieval quality

```bash
python scripts/evaluate.py
```

Requires the pipeline to have already been run (so there are embedded
chunks to search). Writes results to `docs/evaluation.md`. Re-run with a
different `EMBEDDING_PROVIDER` to compare tiers -- see
[evaluation.md](evaluation.md) for the current committed numbers.

## Using the real Hugging Face dataset

If your environment *does* have outbound access to huggingface.co (this
project's own build environment doesn't -- see
[architecture.md](architecture.md)):

```bash
pip install -e ".[datasets]"
export DATA_SOURCE=hf
rag-support pipeline
```

This streams the ~27k-row Bitext customer-support dataset instead of the
80-row bundled sample. Everything downstream (cleaning, embedding,
retrieval, the API, the UI) works identically either way -- `DataSource` is
the one interface that abstracts over "where did this document come from."

## Troubleshooting

- **`psycopg.OperationalError: connection refused`** — Postgres isn't
  running, or `DATABASE_URL` doesn't match how you created the role/database
  in step 2. `pg_lsclusters` (Debian/Ubuntu) or `docker compose ps` shows
  whether the server is up.
- **`relation "documents" does not exist`** — you skipped `rag-support init-db`
  (or `docker compose`'s `api` service hasn't finished its startup migration
  yet — check `docker compose logs api`).
- **`RuntimeError: EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY`** —
  exactly what it says; either set the key or switch back to `auto` /
  `offline` / `local`. Explicit provider modes fail loudly on purpose
  rather than silently falling back -- see [architecture.md](architecture.md).
- **`HuggingFaceSource` raises `RuntimeError` about the Hub** — no outbound
  network access to huggingface.co from your environment. Set
  `DATA_SOURCE=sample` (the default) to use the bundled offline dataset
  instead.
