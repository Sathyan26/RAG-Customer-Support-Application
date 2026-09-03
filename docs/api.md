# API reference

The FastAPI app (`rag_support.api.main:app`) exposes four endpoints.
Interactive, always-up-to-date docs are auto-generated at **`/docs`**
(Swagger UI) and **`/redoc`** whenever the server is running — this file is
a narrative companion to that, with real examples captured while building
and testing the service.

Base URL in every example below: `http://localhost:8000`.

## `GET /health`

Reports which providers are active and how much data is loaded. Cheap,
side-effect-free, safe for a load balancer / uptime check / container
`HEALTHCHECK` (the Dockerfile uses it for exactly that).

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

```json
{
    "status": "ok",
    "database": "ok",
    "embedding_provider": "offline",
    "llm_provider": "offline",
    "document_count": 80,
    "chunk_count": 74
}
```

`status` is `"degraded"` (not an HTTP error) if the database query fails —
the endpoint always returns 200 so it stays useful for diagnosing *why*
something's unhealthy rather than just that it is.

## `POST /chat`

The core endpoint: retrieve relevant knowledge-base chunks, generate a
grounded answer, and persist the turn.

**Request body:**

| Field | Type | Required | Notes |
|---|---|---|---|
| `question` | string | yes | 1-2000 characters |
| `conversation_id` | int \| null | no | omit to start a new conversation; pass a previous response's `conversation_id` to continue one |
| `category` | string \| null | no | restrict retrieval to one category (`ACCOUNT`, `BILLING`, `ORDERS`, `SHIPPING`, `RETURNS`, `TECHNICAL`, `SUBSCRIPTION`, `PRIVACY`, `CONTACT`) |
| `top_k` | int \| null | no | 1-20, overrides the server's default `TOP_K` for this request |

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I cancel my subscription?"}' | python3 -m json.tool
```

Real captured response (offline/offline providers — an OpenAI-backed
response would read as fluent prose instead of the extractive tier's
verbatim-passage-plus-disclaimer format, but the `sources` shape is
identical either way):

```json
{
    "conversation_id": 1,
    "answer": "- (How do I cancel my subscription?) Customer: How do I cancel my subscription? Support: Cancel anytime from Settings > Billing > Manage Plan > Cancel Subscription. You keep access through the end of the current billing period, and no further charges are made. Cancelling doesn't delete your data -- use the separate account-deletion flow for that.\n\n...\n\n_(This answer was returned by the offline extractive fallback...)_",
    "sources": [
        {
            "rank": 1,
            "chunk_id": 24,
            "document_id": 23,
            "source_id": "kb-0019",
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

`sources` is always present and always ranked best-first (`rank: 1` = most
similar). `similarity` is a pgvector cosine similarity clamped to `[0, 1]`
for display. Every source traces back to one `document_chunks` row
(`chunk_id`) and its parent document (`document_id`, `source_id` — the
original external ID from whatever `DataSource` it came from).

**Continuing a conversation:**

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What about two-factor auth?", "conversation_id": 1}'
```

**Errors:**

| Status | Cause |
|---|---|
| 422 | `question` missing/empty, or a field fails validation (e.g. `top_k` out of range) |
| 404 | `conversation_id` doesn't exist |

## `GET /documents`

Lists ingested documents with their chunk counts — useful for inspecting
what's actually in the knowledge base.

**Query params:** `category` (optional), `status` (optional — `ingested`,
`cleaned`, `rejected_duplicate`, `rejected_low_quality`), `limit` (default
50, max 500).

```bash
curl -s "http://localhost:8000/documents?category=BILLING&limit=3" | python3 -m json.tool
```

```json
[
    {
        "id": 1,
        "source": "sample",
        "category": "BILLING",
        "intent": "view_invoice",
        "title": "I need a PDF receipt for last month's charge.",
        "status": "cleaned",
        "chunk_count": 1
    }
]
```

## `POST /ingest`

Runs the full ingest → clean → embed pipeline (whatever `DataSource` and
`EmbeddingProvider` the server is configured with) and returns a summary.
Takes no request body.

```bash
curl -s -X POST http://localhost:8000/ingest | python3 -m json.tool
```

First run against an empty database:

```json
{
    "data_source": "sample",
    "records_read": 80,
    "documents_written": 80,
    "documents_cleaned": 74,
    "duplicates_rejected": 6,
    "low_quality_rejected": 0,
    "chunks_created": 74,
    "chunks_embedded": 74,
    "embedding_provider": "offline"
}
```

Re-running it against the same source (real captured output — this is
idempotency at the *cleaning* stage, not the ingestion stage; see
[architecture.md](architecture.md)'s future-improvements note on
ingestion-time upsert):

```json
{
    "data_source": "sample",
    "records_read": 80,
    "documents_written": 80,
    "documents_cleaned": 0,
    "duplicates_rejected": 80,
    "low_quality_rejected": 0,
    "chunks_created": 0,
    "chunks_embedded": 0,
    "embedding_provider": "offline"
}
```

Every one of the 80 newly-inserted rows is correctly recognized as
content-identical to something already cleaned (via the persisted
`content_hash` column) and rejected before any embedding calls are wasted
on it.

**Note:** this endpoint runs synchronously and blocks until the whole
pipeline finishes. Fine for the 80-row sample dataset (sub-second); for the
full ~27k-row Hugging Face dataset (`DATA_SOURCE=hf`), expect it to take
noticeably longer, proportional to embedding-provider throughput. See
"Future improvements" in [architecture.md](architecture.md).

## CORS

The API allows all origins (`allow_origins=["*"]`) — appropriate for a
demo/portfolio service with no authentication, not for a real multi-tenant
deployment. Restrict `main.py`'s `CORSMiddleware` configuration before
deploying this anywhere with real user data.
