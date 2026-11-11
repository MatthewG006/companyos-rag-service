# CompanyOS RAG Service

A retrieval-augmented generation (RAG) microservice for
[CompanyOS](https://github.com/MatthewG006/CompanyOS), the agent platform that
runs three small businesses (Sky Mountain Cloud, Plenty of Plants, and
Sky Mountain Graphics). It indexes the platform's business knowledge —
project memories, brand kits, support cases, and content — and answers
questions about it with **semantic search and cited, grounded answers**.

Built with **Python, FastAPI, PostgreSQL + pgvector, and LangChain**, with
provider-neutral adapters for embeddings and answer generation (Ollama
open-weight models locally, OpenAI hosted models, or deterministic offline
stand-ins for tests).

## What it does

```
documents ──► chunk (LangChain) ──► embed ──► pgvector (PostgreSQL)
                                                    │
question ──► embed ──► cosine search ◄──────────────┘
              │
              └────► retrieve top-k ──► LLM answer with [n] citations
```

- **Ingest** — documents are split with LangChain's
  `RecursiveCharacterTextSplitter`, embedded, and stored as vectors.
  Re-ingesting a document replaces its chunks, so ingestion is idempotent.
- **Search** — cosine-similarity search over chunks, optionally filtered to
  one business (project).
- **Ask** — a LangChain-composed chain (prompt → LLM → parser) answers from
  the retrieved context only, with citations. When retrieval finds nothing
  usable, the service says so instead of letting the model improvise.
- **CompanyOS sync** — `POST /ingest/companyos` pulls retrievable knowledge
  straight from the CompanyOS database (project memories, brand kits,
  support cases, content items). Customer PII is deliberately not indexed.

## Quickstart

### Offline demo (no database, no models)

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
RAG_EMBEDDER=dev RAG_LLM=dev RAG_EMBED_DIM=256 \
  .venv/bin/uvicorn app.main:app --port 8100
```

```bash
curl -X POST localhost:8100/ingest -H 'Content-Type: application/json' -d '{
  "documents": [{
    "source": "project_memory", "external_id": "fern-care",
    "title": "Fern care guide", "project_id": "plenty-of-plants",
    "text": "Water the ferns twice a week and keep greenhouse humidity near sixty percent."
  }]
}'
curl -X POST localhost:8100/ask -H 'Content-Type: application/json' \
  -d '{"question": "How often should I water the ferns?"}'
```

The `dev` providers are deterministic stand-ins (hashing embeddings, an
extractive answerer) so the whole pipeline — chunking, storage, ranking,
API, auth — can be exercised and tested with zero external services. They
are not semantic models; use Ollama or OpenAI for real retrieval quality.

### Production shape (Docker)

```bash
ollama pull nomic-embed-text && ollama pull llama3.1   # on the host
docker compose up --build
```

This starts PostgreSQL with pgvector and the API on port 8100, wired to the
host's Ollama for embeddings and generation.

## API

| Endpoint | Description |
| --- | --- |
| `GET /health` | Store/provider status. No auth. |
| `POST /ingest` | Ingest `{documents: [{source, external_id, title, text, project_id}]}`. |
| `POST /ingest/companyos` | Ingest knowledge from the CompanyOS database. |
| `POST /search` | `{query, project_id?, limit}` → ranked chunks with cosine scores. |
| `POST /ask` | `{question, project_id?, limit}` → grounded answer + citations. |
| `POST /ask/jobs` | Same body as `/ask`; returns `202 {job_id}` immediately and answers in the background. |
| `GET /ask/jobs/{job_id}` | Poll a job: `queued` → `running` → `done` (answer + citations) or `error`. |

When `RAG_SERVICE_TOKEN` is set, every endpoint except `/health` requires
it as a Bearer token — the same owner-token pattern CompanyOS itself uses.
Interactive OpenAPI docs are served at `/docs`.

## Configuration

All settings are environment variables with the `RAG_` prefix; see
`.env.example` for the full list. Highlights:

- `RAG_DATABASE_URL` — PostgreSQL URL for the pgvector store. Empty uses an
  in-memory store. **Supabase note:** the connection pooler presents a
  certificate chain signed by Supabase's own CA, which stock TLS
  verification rejects before login; append `?sslmode=no-verify` to the URL.
- `RAG_EMBEDDER` / `RAG_LLM` — `ollama` (default), `openai`, or `dev`.
- `RAG_EMBED_DIM` — must match the embedding model (nomic-embed-text: 768,
  text-embedding-3-small: 1536, dev: 256). Switching embedding models means
  re-creating the vector table at the new dimension and re-ingesting.
- `RAG_MIN_SCORE` — cosine floor for `/ask`; hits below it are dropped
  before answering. Tune per embedding model (roughly 0.25–0.4 for real
  models).

## Design notes

- **Provider-neutral, like CompanyOS itself.** Embeddings, the LLM, and the
  vector store each sit behind a small `Protocol` interface, so swapping
  Ollama ↔ OpenAI or pgvector ↔ in-memory changes configuration, not code —
  and the test suite runs entirely on the offline implementations.
- **Grounding discipline.** Answers are generated only from retrieved
  chunks, always returned with their citations, and an empty retrieval is a
  plain "I couldn't find that" rather than a hallucinated answer.
- **Errors don't leak.** Provider and database failures surface as clean
  `502`s with the exception type only; keys and connection strings are
  never logged.

## Development

```bash
.venv/bin/ruff check app tests
.venv/bin/pytest -q
```

CI (GitHub Actions) runs the same lint + tests on every push and PR.

## Roadmap

- LangGraph agent endpoint that can call CompanyOS tools (task creation,
  lead lookup) with retrieval in the loop.
- An MCP server exposing search/ask as tools to CompanyOS agents.
- Hybrid retrieval (pgvector + Postgres full-text) and a small evaluation
  harness (hit-rate and answer-grounding checks) over a golden question set.
