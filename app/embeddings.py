"""Embedding providers behind one small interface.

- OllamaEmbedder: local open-weight embedding models (default nomic-embed-text).
- OpenAIEmbedder: hosted embeddings (default text-embedding-3-small).
- DevEmbedder: deterministic hashing embedder. It is NOT a semantic model;
  it exists so tests and offline smoke runs exercise the full pipeline with
  zero external services. Lexically similar texts get similar vectors, which
  is enough to prove ranking, storage, and API behavior end to end.
"""

import hashlib
import math
import re
from typing import Protocol

import httpx

from app.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""
        ...


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class DevEmbedder:
    """Deterministic bag-of-words hashing embedder for offline development."""

    name = "dev"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _TOKEN_RE.findall(text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            out.append(_normalize(vec))
        return out


class OllamaEmbedder:
    name = "ollama"

    def __init__(self, settings: Settings):
        self._base = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_embed_model
        self.dim = settings.embed_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with httpx.Client(timeout=120.0) as client:
            for text in texts:
                resp = client.post(
                    f"{self._base}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
                resp.raise_for_status()
                vectors.append(resp.json()["embedding"])
        return vectors


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ValueError("RAG embedder is 'openai' but no API key is configured (OPENAI_API_KEY).")
        self._key = settings.openai_api_key
        self._model = settings.openai_embed_model
        self.dim = settings.embed_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._key}"},
                json={"model": self._model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]


def build_embedder(settings: Settings) -> Embedder:
    kind = settings.embedder.lower()
    if kind == "dev":
        return DevEmbedder(dim=settings.embed_dim)
    if kind == "ollama":
        return OllamaEmbedder(settings)
    if kind == "openai":
        return OpenAIEmbedder(settings)
    raise ValueError(f"Unknown embedder provider: {settings.embedder!r} (expected ollama, openai, or dev)")
