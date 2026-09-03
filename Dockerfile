# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is only here for the container HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md alembic.ini ./
COPY src ./src
COPY scripts ./scripts
COPY frontend ./frontend

# openai + datasets are lightweight and cover the "production" provider
# path plus the real Hugging Face data source. local-ml (sentence-
# transformers/transformers/torch) is deliberately left out of the default
# image -- it's a large, optional dependency for the local-model tier; see
# docs/setup.md for building a variant image with it included. frontend
# (streamlit) is included so this one image can run either the API or the
# UI, selected by the command each docker-compose service passes in.
RUN pip install --no-cache-dir ".[openai,datasets,frontend]"

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["rag-support", "serve"]
