"""Answer generation: retrieve -> ground -> generate (a small RAG chain).

The chain is composed with LangChain core runnables: a chat prompt template
renders the retrieved context and question, the configured LLM completes it,
and a string output parser returns the answer. Providers mirror the
embedder pattern — Ollama (local open-weight), OpenAI (hosted), and a
deterministic extractive "dev" model for offline runs and tests.

Grounding discipline: when retrieval finds nothing, the service answers
that it found nothing instead of letting the model improvise.
"""

import re
from typing import Protocol

import httpx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.config import Settings
from app.embeddings import Embedder
from app.store import Hit, VectorStore

NO_MATCH_ANSWER = (
    "I couldn't find anything in the indexed business documents that answers that. "
    "Try ingesting more documents, or rephrasing the question."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about the owner's businesses using only the "
            "retrieved context below. Cite sources inline like [1] and [2]. "
            "If the context does not contain the answer, say so plainly.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


class LLM(Protocol):
    name: str

    def complete(self, prompt: str) -> str:
        ...


class DevLLM:
    """Extractive stand-in: answers by quoting the top retrieved passages.

    Not a language model — it exists so the /ask path can be exercised and
    tested offline. Production deployments use Ollama or OpenAI.
    """

    name = "dev"

    def complete(self, prompt: str) -> str:
        # Only parse the rendered Context block. The system instructions
        # themselves mention "[1]" as a citation example, so a naive scan of
        # the whole prompt would quote the instructions back.
        marker = "Context:\n"
        start = prompt.rfind(marker)
        context = prompt[start + len(marker):] if start >= 0 else ""
        sections = re.findall(r"\[\d+\] [^\n]*\n(.*?)(?=\n\[\d+\] |\Z)", context, re.S)
        if not sections:
            return NO_MATCH_ANSWER
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", sections[0].strip())
            if s.strip() and not s.startswith(("Human:", "System:"))
        ]
        if not sentences:
            return NO_MATCH_ANSWER
        return "Based on the retrieved documents: " + " ".join(sentences[:2])


class OllamaLLM:
    name = "ollama"

    def __init__(self, settings: Settings):
        self._base = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_llm_model

    def complete(self, prompt: str) -> str:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{self._base}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            return resp.json()["response"].strip()


class OpenAILLM:
    name = "openai"

    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ValueError("RAG llm is 'openai' but no API key is configured (OPENAI_API_KEY).")
        self._key = settings.openai_api_key
        self._model = settings.openai_llm_model

    def complete(self, prompt: str) -> str:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._key}"},
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()


def build_llm(settings: Settings) -> LLM:
    kind = settings.llm.lower()
    if kind == "dev":
        return DevLLM()
    if kind == "ollama":
        return OllamaLLM(settings)
    if kind == "openai":
        return OpenAILLM(settings)
    raise ValueError(f"Unknown llm provider: {settings.llm!r} (expected ollama, openai, or dev)")


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        label = hit.title or hit.source
        blocks.append(f"[{i}] {label} ({hit.source}:{hit.external_id})\n{hit.content}")
    return "\n\n".join(blocks)


def answer_question(
    store: VectorStore,
    embedder: Embedder,
    llm: LLM,
    question: str,
    project_id: str | None,
    limit: int,
    min_score: float = 0.0,
) -> tuple[str, list[Hit]]:
    query_vector = embedder.embed([question])[0]
    hits = store.search(query_vector, project_id, limit)
    if min_score > 0:
        hits = [h for h in hits if h.score >= min_score]
    # When retrieval finds nothing usable, decline instead of letting the
    # model improvise an answer with no grounding.
    if not hits:
        return NO_MATCH_ANSWER, []
    chain = PROMPT | RunnableLambda(lambda msg: llm.complete(msg.to_string())) | StrOutputParser()
    answer = chain.invoke({"context": format_context(hits), "question": question})
    return answer, hits
