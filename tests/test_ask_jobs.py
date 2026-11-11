"""Background ask jobs: start returns immediately, polling reaches a result."""

import time

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

PLANTS = {
    "source": "project_memory",
    "external_id": "plants-1",
    "title": "Fern care",
    "text": "Water the ferns twice a week. Keep the greenhouse humidity near sixty percent.",
    "project_id": "plenty-of-plants",
}


def _poll_until_done(client, job_id: str, deadline_seconds: float = 10.0) -> dict:
    deadline = time.time() + deadline_seconds
    while time.time() < deadline:
        resp = client.get(f"/ask/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in ("done", "error"):
            return body
        assert body["status"] in ("queued", "running")
        time.sleep(0.05)
    raise AssertionError("ask job did not finish before the deadline")


def test_ask_job_roundtrip(client):
    client.post("/ingest", json={"documents": [PLANTS]})

    resp = client.post("/ask/jobs", json={"question": "How often should I water the ferns?"})
    assert resp.status_code == 202
    created = resp.json()
    assert created["status"] == "queued"
    assert created["job_id"]

    body = _poll_until_done(client, created["job_id"])
    assert body["status"] == "done"
    assert body["question"] == "How often should I water the ferns?"
    assert "twice a week" in body["answer"]
    assert body["citations"], "job answers must carry citations"
    assert body["citations"][0]["external_id"] == "plants-1"
    assert body["error"] is None


def test_ask_job_unknown_id_is_404(client):
    resp = client.get("/ask/jobs/does-not-exist")
    assert resp.status_code == 404


def test_ask_jobs_require_auth_when_configured():
    settings = Settings(
        embedder="dev", llm="dev", embed_dim=256, database_url="", service_token="test-secret"
    )
    with TestClient(create_app(settings)) as authed_client:
        assert authed_client.post("/ask/jobs", json={"question": "ferns?"}).status_code == 401
        assert authed_client.get("/ask/jobs/abc").status_code == 401
        ok = authed_client.post(
            "/ask/jobs",
            json={"question": "ferns?"},
            headers={"Authorization": "Bearer test-secret"},
        )
        assert ok.status_code == 202
