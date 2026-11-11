"""HTTP request/response models for the RAG service API."""

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    """One source document to ingest."""

    source: str = Field(
        description="Where the document came from, e.g. 'project_memory', 'brand_kit', 'support_case'."
    )
    external_id: str = Field(description="Stable id of the document in its source system.")
    title: str = ""
    text: str = Field(min_length=1)
    project_id: str | None = Field(
        default=None, description="CompanyOS project id the document belongs to, if any."
    )


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1)


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class CompanyOSIngestResponse(BaseModel):
    documents: int
    chunks: int
    sources: dict[str, int]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    project_id: str | None = None
    limit: int = Field(default=5, ge=1, le=25)


class SearchHit(BaseModel):
    source: str
    external_id: str
    title: str
    project_id: str | None
    chunk_index: int
    content: str
    score: float = Field(description="Cosine similarity in [0, 1]; higher is closer.")


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    project_id: str | None = None
    limit: int = Field(default=5, ge=1, le=25)


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[SearchHit]


class HealthResponse(BaseModel):
    status: str
    store: str
    embedder: str
    llm: str
