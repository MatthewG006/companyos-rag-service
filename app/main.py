"""CompanyOS RAG service — FastAPI application.

Endpoints:
- GET  /health            provider and store status (no auth)
- POST /ingest            ingest documents (chunk -> embed -> store)
- POST /ingest/companyos  ingest knowledge straight from the CompanyOS DB
- POST /search            semantic search over ingested chunks
- POST /ask               RAG: retrieve, then answer with citations
- POST /ask/jobs          start an ask in the background, returns a job id
- GET  /ask/jobs/{job_id} poll an ask job (queued -> running -> done/error)
- POST /diagnose/jobs     start a mechanic diagnosis of worker health (background)
- GET  /diagnose/jobs/{job_id} poll a diagnose job

The job endpoints exist so callers behind proxies with a response-wait
limit never hold one request open for a whole CPU generation: starting a
job returns immediately and polling is a series of short requests.

When RAG_SERVICE_TOKEN is set, every endpoint except /health requires it
as a Bearer token — the same owner-token pattern CompanyOS itself uses.
"""

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings
from app.diagnose import diagnose_worker
from app.embeddings import build_embedder
from app.ingest import ingest_documents, read_companyos_documents
from app.rag import answer_question, build_llm
from app.schemas import (
    AskJobCreated,
    AskJobStatus,
    AskRequest,
    AskResponse,
    CompanyOSIngestResponse,
    DiagnoseJobCreated,
    DiagnoseJobStatus,
    DiagnoseRequest,
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

    # --- Background ask jobs -------------------------------------------------
    # In-memory registry: jobs are short-lived polls for a single caller, so
    # no persistence is needed. Entries expire after _ASK_JOB_TTL_SECONDS.
    _ASK_JOB_TTL_SECONDS = 15 * 60
    ask_jobs: dict[str, dict] = {}
    ask_jobs_lock = threading.Lock()
    ask_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ask-job")

    def _run_ask_job(state, job_id: str, body: AskRequest) -> None:
        with ask_jobs_lock:
            if job_id in ask_jobs:
                ask_jobs[job_id]["status"] = "running"
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
        except Exception as exc:  # same surface as the sync endpoint: type name only
            log.exception("ask job %s failed", job_id)
            with ask_jobs_lock:
                if job_id in ask_jobs:
                    ask_jobs[job_id].update(status="error", error=f"Ask failed: {type(exc).__name__}")
            return
        with ask_jobs_lock:
            if job_id in ask_jobs:
                ask_jobs[job_id].update(
                    status="done",
                    answer=answer,
                    citations=[_to_hit_schema(h).model_dump() for h in hits],
                )

    @app.post(
        "/ask/jobs",
        response_model=AskJobCreated,
        status_code=202,
        dependencies=[Depends(require_auth)],
    )
    def start_ask_job(request: Request, body: AskRequest) -> AskJobCreated:
        job_id = uuid.uuid4().hex
        now = time.time()
        with ask_jobs_lock:
            stale = [j for j, job in ask_jobs.items() if now - job["created_at"] > _ASK_JOB_TTL_SECONDS]
            for j in stale:
                ask_jobs.pop(j, None)
            ask_jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "question": body.question,
                "answer": None,
                "citations": None,
                "error": None,
                "created_at": now,
            }
        ask_executor.submit(_run_ask_job, request.app.state, job_id, body)
        return AskJobCreated(job_id=job_id, status="queued")

    @app.get("/ask/jobs/{job_id}", response_model=AskJobStatus, dependencies=[Depends(require_auth)])
    def get_ask_job(job_id: str) -> AskJobStatus:
        with ask_jobs_lock:
            job = ask_jobs.get(job_id)
            snapshot = dict(job) if job is not None else None
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Unknown ask job.")
        return AskJobStatus(
            job_id=snapshot["job_id"],
            status=snapshot["status"],
            question=snapshot["question"],
            answer=snapshot["answer"],
            citations=snapshot["citations"],
            error=snapshot["error"],
        )

    # --- Background diagnose jobs -------------------------------------------
    # Same pattern as ask jobs: a diagnosis is a full CPU generation, so it
    # runs on a background thread and callers poll with short requests.
    _DIAGNOSE_JOB_TTL_SECONDS = 15 * 60
    diagnose_jobs: dict[str, dict] = {}
    diagnose_jobs_lock = threading.Lock()
    diagnose_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diagnose-job")

    def _run_diagnose_job(state, job_id: str, body: DiagnoseRequest) -> None:
        with diagnose_jobs_lock:
            if job_id in diagnose_jobs:
                diagnose_jobs[job_id]["status"] = "running"
        try:
            diagnosis = diagnose_worker(state.llm, body.evidence)
        except Exception as exc:  # same surface as the other endpoints: type name only
            log.exception("diagnose job %s failed", job_id)
            with diagnose_jobs_lock:
                if job_id in diagnose_jobs:
                    diagnose_jobs[job_id].update(
                        status="error", error=f"Diagnose failed: {type(exc).__name__}"
                    )
            return
        with diagnose_jobs_lock:
            if job_id in diagnose_jobs:
                diagnose_jobs[job_id].update(status="done", diagnosis=diagnosis)

    @app.post(
        "/diagnose/jobs",
        response_model=DiagnoseJobCreated,
        status_code=202,
        dependencies=[Depends(require_auth)],
    )
    def start_diagnose_job(request: Request, body: DiagnoseRequest) -> DiagnoseJobCreated:
        job_id = uuid.uuid4().hex
        now = time.time()
        with diagnose_jobs_lock:
            stale = [
                j
                for j, job in diagnose_jobs.items()
                if now - job["created_at"] > _DIAGNOSE_JOB_TTL_SECONDS
            ]
            for j in stale:
                diagnose_jobs.pop(j, None)
            diagnose_jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "diagnosis": None,
                "error": None,
                "created_at": now,
            }
        diagnose_executor.submit(_run_diagnose_job, request.app.state, job_id, body)
        return DiagnoseJobCreated(job_id=job_id, status="queued")

    @app.get(
        "/diagnose/jobs/{job_id}",
        response_model=DiagnoseJobStatus,
        dependencies=[Depends(require_auth)],
    )
    def get_diagnose_job(job_id: str) -> DiagnoseJobStatus:
        with diagnose_jobs_lock:
            job = diagnose_jobs.get(job_id)
            snapshot = dict(job) if job is not None else None
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Unknown diagnose job.")
        return DiagnoseJobStatus(
            job_id=snapshot["job_id"],
            status=snapshot["status"],
            diagnosis=snapshot["diagnosis"],
            error=snapshot["error"],
        )

    return app


app = create_app()
