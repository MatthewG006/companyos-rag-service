import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture()
def settings() -> Settings:
    # Fully offline configuration: dev embedder + dev LLM + in-memory store.
    return Settings(embedder="dev", llm="dev", embed_dim=256, database_url="", service_token="")


@pytest.fixture()
def client(settings: Settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client
