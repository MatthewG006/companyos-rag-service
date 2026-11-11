"""Vector stores behind one small interface.

- PgVectorStore: production store. Embeddings live in PostgreSQL via the
  pgvector extension, in the same database family CompanyOS already uses.
- InMemoryStore: development store with identical ranking semantics (cosine
  similarity), used when no database URL is configured and by the tests.
"""

import math
from dataclasses import dataclass, field
from typing import Protocol

from app.config import Settings


@dataclass
class Chunk:
    source: str
    external_id: str
    title: str
    project_id: str | None
    chunk_index: int
    content: str
    embedding: list[float] = field(repr=False)


@dataclass
class Hit:
    source: str
    external_id: str
    title: str
    project_id: str | None
    chunk_index: int
    content: str
    score: float


class VectorStore(Protocol):
    name: str

    def replace_document(self, chunks: list[Chunk]) -> None:
        """Atomically replace all stored chunks for the chunks' document."""
        ...

    def search(self, query_embedding: list[float], project_id: str | None, limit: int) -> list[Hit]:
        ...

    def count(self) -> int:
        ...


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class InMemoryStore:
    name = "memory"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def replace_document(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        first = chunks[0]
        self._chunks = [
            c
            for c in self._chunks
            if not (c.source == first.source and c.external_id == first.external_id)
        ]
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], project_id: str | None, limit: int) -> list[Hit]:
        scored = [
            Hit(c.source, c.external_id, c.title, c.project_id, c.chunk_index, c.content,
                _cosine(query_embedding, c.embedding))
            for c in self._chunks
            if project_id is None or c.project_id == project_id
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]

    def count(self) -> int:
        return len(self._chunks)


class PgVectorStore:
    """pgvector-backed store. The table is created on first use.

    The embedding dimension is fixed at table creation from the configured
    embedder; switching embedding models means re-creating the table and
    re-ingesting, which the /ingest endpoints make a one-call job.
    """

    name = "pgvector"

    def __init__(self, settings: Settings):
        import psycopg
        from pgvector.psycopg import register_vector

        self._table = settings.table_name
        self._dim = settings.embed_dim
        self._conn = psycopg.connect(settings.database_url, autocommit=True)
        with self._conn.cursor() as cur:
            # The extension must exist before register_vector looks up the
            # vector type — on a fresh database the order matters.
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id BIGSERIAL PRIMARY KEY,
                    source TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    project_id TEXT,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({self._dim}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_doc_idx "
                f"ON {self._table} (source, external_id)"
            )

    def replace_document(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        first = chunks[0]
        with self._conn.transaction():
            with self._conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self._table} WHERE source = %s AND external_id = %s",
                    (first.source, first.external_id),
                )
                cur.executemany(
                    f"INSERT INTO {self._table} "
                    "(source, external_id, title, project_id, chunk_index, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [
                        (
                            c.source, c.external_id, c.title, c.project_id,
                            c.chunk_index, c.content, c.embedding,
                        )
                        for c in chunks
                    ],
                )

    def search(self, query_embedding: list[float], project_id: str | None, limit: int) -> list[Hit]:
        sql = (
            f"SELECT source, external_id, title, project_id, chunk_index, content, "
            f"1 - (embedding <=> %s::vector) AS score FROM {self._table} "
        )
        params: list = [query_embedding]
        if project_id is not None:
            sql += "WHERE project_id = %s "
            params.append(project_id)
        sql += "ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, limit])
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            Hit(r[0], r[1], r[2], r[3], r[4], r[5], float(r[6])) for r in rows
        ]

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            return int(cur.fetchone()[0])


def build_store(settings: Settings) -> VectorStore:
    if settings.database_url:
        return PgVectorStore(settings)
    return InMemoryStore()
