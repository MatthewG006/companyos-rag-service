from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag import NO_MATCH_ANSWER

PLANTS = {
    "source": "project_memory",
    "external_id": "plants-1",
    "title": "Fern care",
    "text": "Water the ferns twice a week. Keep the greenhouse humidity near sixty percent.",
    "project_id": "plenty-of-plants",
}
PRICING = {
    "source": "brand_kit",
    "external_id": "web-1",
    "title": "Services",
    "text": "Sky Mountain Cloud builds websites for local service businesses with monthly care plans.",
    "project_id": "sky-mountain-cloud",
}


def test_health_reports_dev_providers(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok", "store": "memory", "embedder": "dev", "llm": "dev"}


def test_ingest_search_ask_roundtrip(client):
    resp = client.post("/ingest", json={"documents": [PLANTS, PRICING]})
    assert resp.status_code == 200
    assert resp.json()["documents"] == 2
    assert resp.json()["chunks"] >= 2

    resp = client.post("/search", json={"query": "how often should I water the ferns", "limit": 3})
    assert resp.status_code == 200
    hits = resp.json()["hits"]
    assert hits, "expected at least one hit"
    assert hits[0]["external_id"] == "plants-1"

    resp = client.post("/ask", json={"question": "How often should I water the ferns?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "twice a week" in body["answer"]
    assert body["citations"], "answers must carry citations"
    assert body["citations"][0]["external_id"] == "plants-1"


def test_search_can_filter_by_project(client):
    client.post("/ingest", json={"documents": [PLANTS, PRICING]})
    resp = client.post(
        "/search",
        json={"query": "websites for local businesses", "project_id": "sky-mountain-cloud"},
    )
    hits = resp.json()["hits"]
    assert hits and all(h["project_id"] == "sky-mountain-cloud" for h in hits)


def test_ask_with_empty_store_declines_to_invent(client):
    resp = client.post("/ask", json={"question": "What is the pricing for logo design?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == NO_MATCH_ANSWER
    assert body["citations"] == []


def test_auth_token_enforced_when_configured():
    settings = Settings(
        embedder="dev", llm="dev", embed_dim=256, database_url="", service_token="test-secret"
    )
    with TestClient(create_app(settings)) as authed_client:
        assert authed_client.get("/health").status_code == 200, "health stays open"
        assert authed_client.post("/search", json={"query": "ferns"}).status_code == 401
        ok = authed_client.post(
            "/search",
            json={"query": "ferns"},
            headers={"Authorization": "Bearer test-secret"},
        )
        assert ok.status_code == 200


def test_companyos_ingest_requires_a_configured_database(client):
    resp = client.post("/ingest/companyos")
    assert resp.status_code == 400
