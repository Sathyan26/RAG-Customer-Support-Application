<!-- markdownlint-disable MD033 -->
# RAG Support

A Retrieval-Augmented Generation customer support assistant, built as **one
cohesive application** rather than a pile of standalone scripts and
notebooks. Data ingestion, cleaning, vector storage, retrieval, and
generation all live behind a single pipeline, a single FastAPI service, and
a single CLI.

[![CI](https://github.com/YOUR_GITHUB_USERNAME/rag-customer-support/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_GITHUB_USERNAME/rag-customer-support/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-informational)

## What this is

Northwind Cloud is a fictional SaaS product invented for this project. Its
"customer support knowledge base" — account, billing, orders, shipping,
returns, technical, subscription, privacy, and contact questions — is what
the assistant is grounded in. Ask it a support question and it will:

1. embed your question,
2. retrieve the most relevant knowledge-base passages from Postgres via
   pgvector cosine similarity,
3. hand those passages to an LLM with an explicit "answer only from this
   context, cite your sources" instruction, and
4. return the answer plus exactly which passages it used — never a
   free-floating claim with no traceable source.

The point of the project is not the fictional product; it's the pipeline
underneath it — ingestion, cleaning, storage, embedding, retrieval, and
generation as one system, engineered and tested the way a production
service would be, with every provider swappable and every stage able to
run with zero API keys.

## Architecture

```mermaid
flowchart LR
    subgraph Data["Data pipeline (rag_support.rag.pipeline.run_ingest_clean_embed)"]
        direction LR
        DS["DataSource\n(HF Bitext dataset\nor bundled sample)"] --> ING["Ingest\ndocuments table"]
        ING --> CLN["Clean\nnormalize · dedup · chunk"]
        CLN --> EMB["Embed\npluggable provider"]
    end

    EMB --> PG[(Postgres\n+ pgvector)]

    subgraph Serve["Serving (rag_support.rag.pipeline.RAGPipeline)"]
        direction LR
        Q["User question"] --> RET["Retriever\ncosine similarity search"]
        RET --> PROMPT["Grounded prompt\n+ numbered context"]
        PROMPT --> LLM["LLM provider\nOpenAI · local · offline"]
        LLM --> ANS["Answer + cited sources"]
    end

    PG -.-> RET

    API["FastAPI (/chat /ingest /documents /health)"] --> Serve
    CLI["Typer CLI (rag-support ...)"] --> Data
    CLI --> Serve
    UI["Streamlit chat UI"] --> API
```

Every provider-shaped box above is pluggable behind a small interface, with
an `auto` mode that degrades gracefully:

| Layer | Tiers (best → always-available) | Selected by |
|---|---|---|
| Embeddings | OpenAI `text-embedding-3-small` → local `sentence-transformers/all-MiniLM-L6-v2` → offline `HashingVectorizer` | `EMBEDDING_PROVIDER` |
| Generation | OpenAI `gpt-4o-mini` → local `google/flan-t5-base` → offline extractive fallback | `LLM_PROVIDER` |
| Data source | Hugging Face Bitext customer-support dataset (~27k rows) → bundled offline sample (80 rows) | `DATA_SOURCE` |

The offline tiers exist because this project was built and CI-tested in an
environment with **no outbound network access to huggingface.co** (see
`docs/data_pipeline.md`), and because a portfolio project that only runs
with a paid API key isn't actually runnable by whoever's reviewing it.
Every part of the pluggable design is real production practice, not a
workaround invented for that constraint — graceful provider fallback,
a documented "what changes if I swap this" story, and a system that's
honest about the quality/cost trade-off at each tier.

See **[docs/architecture.md](docs/architecture.md)** for the full design
rationale, database schema, and the trade-offs made along the way.

## Quickstart

### Option A: Docker Compose (recommended)

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/rag-customer-support.git
cd rag-customer-support
docker compose up --build
```

This starts Postgres+pgvector, runs migrations, ingests the bundled sample
knowledge base, and serves the API on **http://localhost:8000** (interactive
docs at `/docs`) and the chat UI on **http://localhost:8501** — no API keys
required; it runs on the offline embedding/LLM tiers by default. To use
OpenAI instead, set `OPENAI_API_KEY` and `EMBEDDING_PROVIDER=openai` /
`LLM_PROVIDER=openai` in a `.env` file before starting.

### Option B: Local Python

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai,datasets,frontend]"

# Needs a local Postgres 16+ with the pgvector extension available --
# see docs/setup.md for exact commands (apt / Homebrew / Docker-just-for-db).
export DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5432/rag_support

rag-support init-db     # create the extension + run Alembic migrations
rag-support pipeline    # ingest -> clean -> embed the sample dataset
rag-support serve       # http://localhost:8000
# in another terminal:
rag-support chat        # interactive terminal chat, no UI needed
```

Full walkthrough, every environment variable, and troubleshooting notes:
**[docs/setup.md](docs/setup.md)**.

## Example

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I cancel my subscription?"}' | python3 -m json.tool
```

```json
{
  "conversation_id": 1,
  "answer": "Cancel anytime from Settings > Billing > Manage Plan > Cancel Subscription. You keep access through the end of the current billing period, and no further charges are made. [1]",
  "sources": [
    {
      "rank": 1,
      "category": "BILLING",
      "title": "How do I cancel my subscription?",
      "similarity": 0.4789,
      "excerpt": "Customer: How do I cancel my subscription? Support: Cancel anytime from Settings > Billing..."
    }
  ],
  "embedding_provider": "offline",
  "llm_provider": "offline"
}
```

(This example is trimmed for the README; a real response also includes the
other retrieved sources. Full endpoint reference:
**[docs/api.md](docs/api.md)**.)

## Retrieval quality

`scripts/evaluate.py` runs a held-out set of 36 hand-written questions (one
per intent, phrased differently from the knowledge-base text itself) through
the retriever and reports Hit@K and Mean Reciprocal Rank. Current committed
result, on the **offline** (lexical hashing) tier — the weakest of the three
and the one that needs no API key or model download:

- **Hit@4**: 86.1%
- **MRR**: 0.750
- **Avg latency**: ~2-4ms per query

Full per-query breakdown and instructions for comparing tiers:
**[docs/evaluation.md](docs/evaluation.md)**.

## Testing

```bash
make test          # or: pytest -v
make lint           # ruff check src tests
```

The 50-test suite runs against a real Postgres + pgvector database (not
mocks, not SQLite — pgvector's type and cosine-distance queries aren't
meaningfully testable any other way) using the offline provider tiers, so it
needs no API keys and no network. See `docs/setup.md` for provisioning the
test database, and `.github/workflows/ci.yml` for how CI does the same
thing with a service container.

## Project layout

```
src/rag_support/
├── config.py              # Settings (env-driven, provider selection)
├── data/                  # Ingestion: DataSource interface + sources
│   ├── sources/           #   HuggingFaceSource, LocalSampleSource
│   └── sample_dataset/    #   bundled offline knowledge base (JSONL)
├── cleaning/              # normalize, chunk, and the cleaning pipeline stage
├── storage/                # SQLAlchemy models, Alembic migrations, repository (DAO)
├── embeddings/             # EmbeddingProvider: openai / local / offline + factory
├── llm/                    # LLMProvider: openai / local / offline + factory + prompts
├── rag/                    # Retriever + RAGPipeline (the single orchestrator)
├── api/                    # FastAPI app, routers, schemas
└── cli.py                  # Typer CLI -- every pipeline stage as a command

frontend/streamlit_app.py   # thin chat UI over the API
scripts/                    # dataset generator, evaluation harness
data/eval/                  # held-out evaluation query set
tests/                       # unit + integration tests (real DB, offline providers)
docs/                        # architecture, setup, API reference, data pipeline, evaluation
```

## Documentation

- **[docs/architecture.md](docs/architecture.md)** — design rationale, schema, trade-offs
- **[docs/setup.md](docs/setup.md)** — step-by-step setup for every path (Docker, local, tests)
- **[docs/api.md](docs/api.md)** — full HTTP API reference with real examples
- **[docs/data_pipeline.md](docs/data_pipeline.md)** — how ingestion/cleaning/embedding actually work
- **[docs/evaluation.md](docs/evaluation.md)** — retrieval evaluation results

## License

MIT — see [LICENSE](LICENSE).
