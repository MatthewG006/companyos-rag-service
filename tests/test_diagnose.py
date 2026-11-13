"""Mechanic diagnosis: strict-JSON parsing, validation, and the job endpoints."""

import time

from app.diagnose import diagnose_worker, parse_diagnosis


class _StubLLM:
    name = "stub"

    def __init__(self, reply: str):
        self.reply = reply

    def complete(self, prompt: str) -> str:
        return self.reply


def test_parse_accepts_allowlisted_repair():
    raw = (
        'Here is my diagnosis: {"summary": "Worker cannot load its Codex config.", '
        '"likely_cause": "config rejected by the CLI", "recommended_action": "code_fix", '
        '"commit": null, "rationale": "Config errors need a code change, not a restart."}'
    )
    result = parse_diagnosis(raw)
    assert result["recommended_action"] == "code_fix"
    assert result["summary"].startswith("Worker cannot load")


def test_parse_rejects_unknown_action():
    raw = (
        '{"summary": "s", "likely_cause": "c", "recommended_action": "delete_everything", '
        '"commit": null, "rationale": "r"}'
    )
    assert parse_diagnosis(raw)["recommended_action"] == "none"


def test_parse_rollback_requires_valid_commit():
    no_commit = (
        '{"summary": "s", "likely_cause": "c", "recommended_action": "rollback_worker", '
        '"commit": null, "rationale": "r"}'
    )
    assert parse_diagnosis(no_commit)["recommended_action"] == "none"
    bad_commit = (
        '{"summary": "s", "likely_cause": "c", "recommended_action": "rollback_worker", '
        '"commit": "rm -rf /", "rationale": "r"}'
    )
    assert parse_diagnosis(bad_commit)["recommended_action"] == "none"
    good = (
        '{"summary": "s", "likely_cause": "c", "recommended_action": "rollback_worker", '
        '"commit": "8f7df2e", "rationale": "r"}'
    )
    result = parse_diagnosis(good)
    assert result["recommended_action"] == "rollback_worker"
    assert result["commit"] == "8f7df2e"


def test_parse_garbage_falls_back():
    result = parse_diagnosis("the worker seems sad today")
    assert result["recommended_action"] == "none"
    assert result["commit"] is None


def test_diagnose_worker_uses_llm_reply():
    llm = _StubLLM(
        '{"summary": "Healthy.", "likely_cause": "none", "recommended_action": "none", '
        '"commit": null, "rationale": "All checks pass."}'
    )
    result = diagnose_worker(llm, {"heartbeat": {"status": "online"}})
    assert result["summary"] == "Healthy."


def _poll_until_done(client, job_id: str, deadline_seconds: float = 10.0) -> dict:
    deadline = time.time() + deadline_seconds
    while time.time() < deadline:
        resp = client.get(f"/diagnose/jobs/{job_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] in ("done", "error"):
            return body
        assert body["status"] in ("queued", "running")
        time.sleep(0.05)
    raise AssertionError("diagnose job did not finish before the deadline")


def test_diagnose_job_roundtrip_with_dev_llm(client):
    # The dev LLM is extractive, not a diagnostician: the job must still
    # complete, with the defensive fallback diagnosis (action "none").
    started = client.post("/diagnose/jobs", json={"evidence": {"heartbeat": {"status": "degraded"}}})
    assert started.status_code == 202
    body = _poll_until_done(client, started.json()["job_id"])
    assert body["status"] == "done"
    assert body["diagnosis"]["recommended_action"] == "none"


def test_diagnose_job_unknown_id_404s(client):
    resp = client.get("/diagnose/jobs/nope")
    assert resp.status_code == 404
