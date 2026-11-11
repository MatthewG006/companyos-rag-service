"""Service configuration, read from the environment (prefix RAG_).

Design notes:
- Providers are pluggable on purpose. Embeddings and answer generation each
  sit behind a small interface with Ollama (local, open-weight models),
  OpenAI (hosted frontier models), and a deterministic offline "dev"
  implementation used by the test suite and local smoke runs.
- The vector store is pgvector in PostgreSQL when RAG_DATABASE_URL is set,
  and an in-memory store otherwise, so the full pipeline runs anywhere.
"""

import os

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Vector store. When database_url is empty the service uses the
    # in-memory development store instead of pgvector.
    database_url: str = ""
    table_name: str = "rag_chunks"

    # Source database for POST /ingest/companyos. Falls back to
    # database_url, which is the normal production setup: the RAG tables
    # live in the same PostgreSQL database as CompanyOS itself.
    companyos_database_url: str = ""

    # Providers: "ollama" | "openai" | "dev"
    embedder: str = "ollama"
    llm: str = "ollama"

    # Ollama (local / self-hosted open-weight models)
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_llm_model: str = "llama3.1"

    # OpenAI (hosted). The key is read from OPENAI_API_KEY, never logged.
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4o-mini"

    # Embedding vector size. Must match the chosen embedding model
    # (nomic-embed-text: 768, text-embedding-3-small: 1536, dev: 256).
    embed_dim: int = 768

    # Chunking (LangChain RecursiveCharacterTextSplitter)
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Retrieval quality floor for /ask: hits with cosine similarity below
    # this are dropped before answering. 0.0 disables the floor. Useful
    # values depend on the embedding model (real models: roughly 0.25-0.4).
    min_score: float = 0.0

    # Optional bearer token for the API. When empty the API is open, which
    # is only appropriate for a trusted local network.
    service_token: str = ""

    @model_validator(mode="after")
    def _fill_openai_key_from_plain_env(self) -> "Settings":
        # RAG_OPENAI_API_KEY wins; the conventional OPENAI_API_KEY is the fallback.
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        return self


def get_settings() -> Settings:
    return Settings()
