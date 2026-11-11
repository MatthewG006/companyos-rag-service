"""Ingest pipeline: documents -> chunks -> embeddings -> vector store.

Chunking uses LangChain's RecursiveCharacterTextSplitter, which splits on
paragraph/sentence boundaries before falling back to characters, so chunks
stay semantically coherent. Re-ingesting a document replaces its chunks
(delete + insert in the store), which keeps the index idempotent.
"""

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.embeddings import Embedder
from app.schemas import DocumentIn
from app.store import Chunk, VectorStore


def chunk_text(text: str, settings: Settings) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def ingest_documents(
    store: VectorStore,
    embedder: Embedder,
    documents: list[DocumentIn],
    settings: Settings,
) -> tuple[int, int]:
    """Ingest documents; returns (document_count, chunk_count)."""
    total_chunks = 0
    for doc in documents:
        parts = chunk_text(doc.text, settings)
        if not parts:
            continue
        vectors = embedder.embed(parts)
        chunks = [
            Chunk(
                source=doc.source,
                external_id=doc.external_id,
                title=doc.title,
                project_id=doc.project_id,
                chunk_index=i,
                content=part,
                embedding=vector,
            )
            for i, (part, vector) in enumerate(zip(parts, vectors, strict=True))
        ]
        store.replace_document(chunks)
        total_chunks += len(chunks)
    return len(documents), total_chunks


def _json_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def read_companyos_documents(database_url: str) -> tuple[list[DocumentIn], dict[str, int]]:
    """Pull retrievable knowledge straight from the CompanyOS database.

    Sources: project memories, brand kits, support cases, and content items.
    Customer PII (contacts, leads) is deliberately not indexed by default.
    """
    import psycopg

    docs: list[DocumentIn] = []
    counts: dict[str, int] = {}

    def add(source: str, doc: DocumentIn) -> None:
        docs.append(doc)
        counts[source] = counts.get(source, 0) + 1

    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT project_id::text, title, content FROM project_memories")
        for project_id, title, content in cur.fetchall():
            add("project_memory", DocumentIn(
                source="project_memory", external_id=project_id, title=title or "",
                text=content or "", project_id=project_id))

        cur.execute(
            "SELECT project_id::text, website_url, phone, service_area, tagline, "
            "colors::text, services::text, instagram_username FROM brand_kits"
        )
        for row in cur.fetchall():
            project_id, website, phone, area, tagline, colors, services, instagram = row
            lines = [
                f"Website: {website}" if website else "",
                f"Tagline: {tagline}" if tagline else "",
                f"Phone: {phone}" if phone else "",
                f"Service area: {area}" if area else "",
                f"Services: {_json_text(services)}" if services else "",
                f"Brand colors: {_json_text(colors)}" if colors else "",
                f"Instagram: @{instagram}" if instagram else "",
            ]
            text = "\n".join(line for line in lines if line)
            if text:
                add("brand_kit", DocumentIn(
                    source="brand_kit", external_id=project_id, title="Brand kit",
                    text=text, project_id=project_id))

        cur.execute(
            "SELECT id::text, project_id::text, title, description, status FROM support_cases"
        )
        for case_id, project_id, title, description, status in cur.fetchall():
            text = f"{title}\n\n{description or ''}\n\nStatus: {status}".strip()
            add("support_case", DocumentIn(
                source="support_case", external_id=case_id, title=title or "",
                text=text, project_id=project_id))

        cur.execute(
            "SELECT id::text, project_id::text, title, caption, status FROM content_items"
        )
        for item_id, project_id, title, caption, status in cur.fetchall():
            text = f"{title}\n\n{caption or ''}\n\nStatus: {status}".strip()
            add("content_item", DocumentIn(
                source="content_item", external_id=item_id, title=title or "",
                text=text, project_id=project_id))

    return docs, counts
