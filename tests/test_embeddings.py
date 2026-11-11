import math

from app.embeddings import DevEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_dev_embedder_is_deterministic_and_normalized():
    embedder = DevEmbedder(dim=256)
    first = embedder.embed(["water the ferns"])[0]
    second = embedder.embed(["water the ferns"])[0]
    assert first == second
    assert len(first) == 256
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-6)


def test_dev_embedder_ranks_lexically_similar_text_closer():
    embedder = DevEmbedder(dim=256)
    query, similar, unrelated = embedder.embed(
        [
            "how often should I water the ferns",
            "water the ferns twice every week",
            "quarterly tax invoices and payroll",
        ]
    )
    assert _cosine(query, similar) > _cosine(query, unrelated)
