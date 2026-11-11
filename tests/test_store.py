from app.embeddings import DevEmbedder
from app.store import Chunk, InMemoryStore


def _doc_chunks(embedder: DevEmbedder, source: str, external_id: str, project_id, texts: list[str]):
    vectors = embedder.embed(texts)
    return [
        Chunk(source=source, external_id=external_id, title="", project_id=project_id,
              chunk_index=i, content=text, embedding=vec)
        for i, (text, vec) in enumerate(zip(texts, vectors, strict=True))
    ]


def test_memory_store_search_ranks_and_filters():
    embedder = DevEmbedder(dim=256)
    store = InMemoryStore()
    store.replace_document(_doc_chunks(embedder, "doc", "plants", "p1",
                                       ["Water the ferns twice a week and mist the leaves."]))
    store.replace_document(_doc_chunks(embedder, "doc", "pricing", "p2",
                                       ["Website design packages start at a fixed monthly rate."]))
    query = embedder.embed(["how often do I water ferns"])[0]

    hits = store.search(query, None, 5)
    assert hits[0].external_id == "plants"
    assert hits[0].score > hits[1].score

    filtered = store.search(query, "p2", 5)
    assert [h.external_id for h in filtered] == ["pricing"]


def test_memory_store_replace_is_idempotent():
    embedder = DevEmbedder(dim=256)
    store = InMemoryStore()
    store.replace_document(_doc_chunks(embedder, "doc", "a", None, ["first version of the text"]))
    store.replace_document(_doc_chunks(embedder, "doc", "a", None, ["second version of the text"]))
    assert store.count() == 1
    hits = store.search(embedder.embed(["second version"])[0], None, 5)
    assert hits[0].content == "second version of the text"
