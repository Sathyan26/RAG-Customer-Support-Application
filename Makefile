.PHONY: install dev-db init-db ingest clean embed pipeline serve chat test lint fmt docker-up docker-down docker-logs

install:      ## Install the package + dev/openai/datasets/frontend extras into the active venv
	pip install -e ".[dev,openai,datasets,frontend]"

dev-db:       ## Start just the Postgres+pgvector container (for running the app/tests locally, outside Docker)
	docker compose up -d db

init-db:      ## Create the pgvector extension and run Alembic migrations
	rag-support init-db

ingest:       ## Run only the ingestion stage
	rag-support ingest

clean:        ## Run only the cleaning stage
	rag-support clean

embed:        ## Run only the embedding stage
	rag-support embed

pipeline:     ## Run ingest + clean + embed as one operation
	rag-support pipeline

serve:        ## Run the API locally with autoreload
	rag-support serve --reload

chat:         ## Interactive terminal chat against the pipeline
	rag-support chat

test:         ## Run the test suite (needs TEST_DATABASE_URL or the default local rag_support_test db)
	pytest -q

lint:         ## Check style/lint rules
	ruff check src tests

fmt:          ## Auto-fix what ruff can
	ruff check --fix src tests

docker-up:    ## Build and start the full stack (db + api + streamlit frontend)
	docker compose up --build

docker-down:  ## Stop the stack and remove its volumes
	docker compose down -v

docker-logs:  ## Tail logs from all services
	docker compose logs -f
