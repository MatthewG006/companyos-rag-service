"""CompanyOS RAG service — FastAPI application.

Endpoints:
- GET  /health            provider and store status (no auth)
- POST /ingest            ingest documents (chunk -> embed -> store)
- POST /ingest/companyos  ingest knowledge straight from the CompanyOS DB
- POST /search            semantic search over ingested chunks
- POST /ask               RAG: retrieve, then answer with citations

When RAG_SERVICE_TOKEN is set, every endpoint except /health requires it
as a Bearer token — the same owner-token pattern CompanyOS itself uses.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings
from app.embeddings import build_embedder
from app.ingest import ingest_documents, read_companyos_documents
from app.rag import answer_question, build_llm
from app.schemas import (
    AskRequest,
    AskResponse,
    CompanyOSIngestResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.store import Hit, build_store

log = logging.getLogger("companyos-rag")
logging.basicConfig(level=logging.INFO)

_bearer = HTTPBearer(auto_error=False)


def _to_hit_schema(hit: Hit) -> SearchHit:
    return SearchHit(
        source=hit.source,
        external_id=hit.external_id,
        title=hit.title,
        project_id=hit.project_id,
        chunk_index=hit.chunk_index,
        content=hit.content,
        score=round(hit.score, 4),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    app.state.embedder = build_embedder(settings)
    app.state.store = build_store(settings)
    app.state.llm = build_llm(settings)
    log.info(
        "RAG service up: store=%s embedder=%s llm=%s auth=%s",
        app.state.store.name,
        app.state.embedder.name,
        app.state.llm.name,
        "token" if settings.service_token else "open",
    )
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="CompanyOS RAG Service",
        description="Semantic search and grounded Q&A over CompanyOS business documents.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings or get_settings()

    def require_auth(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> None:
        token = request.app.state.settings.service_token
        if not token:
            return
        if credentials is None or credentials.credentials != token:
            raise HTTPException(status_code=401, detail="Missing or invalid service token.")

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        state = request.app.state
        return HealthResponse(
            status="ok",
            store=state.store.name,
            embedder=state.embedder.name,
            llm=state.llm.name,
        )

    @app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_auth)])
    def ingest(request: Request, body: IngestRequest) -> IngestResponse:
        state = request.app.state
        try:
            documents, chunks = ingest_documents(state.store, state.embedder, body.documents, state.settings)
        except Exception as exc:  # provider/store failures surface as 502, never as a stack trace
            log.exception("ingest failed")
            raise HTTPException(status_code=502, detail=f"Ingest failed: {type(exc).__name__}") from exc
        return IngestResponse(documents=documents, chunks=chunks)

    @app.post(
        "/ingest/companyos",
        response_model=CompanyOSIngestResponse,
        dependencies=[Depends(require_auth)],
    )
    def ingest_companyos(request: Request) -> CompanyOSIngestResponse:
        state = request.app.state
        settings: Settings = state.settings
        source_url = settings.companyos_database_url or settings.database_url
        if not source_url:
            raise HTTPException(
                status_code=400,
                detail="No CompanyOS database configured "
                "(set RAG_COMPANYOS_DATABASE_URL or RAG_DATABASE_URL).",
            )
        try:
            documents, counts = read_companyos_documents(source_url)
            doc_count, chunk_count = ingest_documents(state.store, state.embedder, documents, settings)
        except Exception as exc:
            log.exception("companyos ingest failed")
            detail = f"CompanyOS ingest failed: {type(exc).__name__}"
            raise HTTPException(status_code=502, detail=detail) from exc
        return CompanyOSIngestResponse(documents=doc_count, chunks=chunk_count, sources=counts)

    @app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_auth)])
    def search(request: Request, body: SearchRequest) -> SearchResponse:
        state = request.app.state
        try:
            vector = state.embedder.embed([body.query])[0]
            hits = state.store.search(vector, body.project_id, body.limit)
        except Exception as exc:
            log.exception("search failed")
            raise HTTPException(status_code=502, detail=f"Search failed: {type(exc).__name__}") from exc
        return SearchResponse(query=body.query, hits=[_to_hit_schema(h) for h in hits])

    @app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_auth)])
    def ask(request: Request, body: AskRequest) -> AskResponse:
        state = request.app.state
        try:
            answer, hits = answer_question(
                state.store,
                state.embedder,
                state.llm,
                body.question,
                body.project_id,
                body.limit,
                min_score=state.settings.min_score,
            )
        except Exception as exc:
            log.exception("ask failed")
            raise HTTPException(status_code=502, detail=f"Ask failed: {type(exc).__name__}") from exc
        return AskResponse(
            question=body.question,
            answer=answer,
            citations=[_to_hit_schema(h) for h in hits],
        )

    return app


app = create_app()
