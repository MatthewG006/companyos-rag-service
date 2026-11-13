"""Mechanic diagnosis: read worker-health evidence, propose a repair.

This is the "different engine" half of the CompanyOS repair design: the
Codex worker cannot diagnose its own substrate failures (it dies before
any model loads), so diagnosis runs here, on this service's LLM (Llama via
Ollama in production). The output is a bounded, structured proposal — the
model may only recommend one of a short allowlist of runtime repairs, a
code fix (which still ships through a pull request and the owner's
merge), or no action. CompanyOS re-validates everything before anything
reaches the owner as an approval, and again before anything executes.
"""

import json
import re

from app.rag import LLM

REPAIR_ACTIONS = ("restart_worker", "update_worker", "rollback_worker")
RECOMMENDED_ACTIONS = REPAIR_ACTIONS + ("code_fix", "none")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{7,40}$")

PROMPT_TEMPLATE = """You are the mechanic for CompanyOS, a multi-agent system whose tasks are executed by a \
Node worker process that runs the Muse. The worker periodically self-checks its substrate \
(Codex config loads, CLI authenticated, database schema complete, OpenAI key present) and reports \
the result in a heartbeat. You are given that evidence as JSON: the freshest heartbeat, recent \
failed runs, and queued run counts.

Diagnose the worker's health from the evidence and answer with ONLY a JSON object — no prose, \
no markdown — with exactly these fields:
- "summary": one or two plain sentences describing the worker's state.
- "likely_cause": the most probable root cause, or "none" if the worker looks healthy.
- "recommended_action": one of "restart_worker" (worker process is stuck or stale; a restart \
clears it), "update_worker" (worker code is behind the latest main; fast-forward and restart), \
"rollback_worker" (the current worker commit is broken; return to the last known-good commit), \
"code_fix" (the fault is in the code or configuration itself; a developer must ship a fix through \
a pull request), "none" (healthy, or no safe action fits).
- "commit": for "rollback_worker" only, the last known-good commit from the evidence; otherwise null.
- "rationale": one sentence on why this action fits the evidence.

Safety rules: only recommend a runtime repair when the evidence supports it. If the Codex config \
is rejected (codex_config_ok is false, or a failed run shows a config-loading error), the fault \
is configuration/code — recommend "code_fix", never a restart loop. If the evidence is empty or \
inconclusive, recommend "none".

Evidence:
{evidence}
"""


def fallback_diagnosis(note: str) -> dict:
    """A safe diagnosis used whenever the model's output cannot be trusted."""
    return {
        "summary": note,
        "likely_cause": "unknown",
        "recommended_action": "none",
        "commit": None,
        "rationale": "No structured recommendation was available; a human should review the evidence.",
    }


def parse_diagnosis(raw: str) -> dict:
    """Extract and validate the model's JSON; anything off-spec degrades
    to the fallback rather than propagating into an owner approval."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return fallback_diagnosis("The mechanic returned no structured diagnosis.")
    try:
        parsed = json.loads(raw[start : end + 1])
    except (ValueError, TypeError):
        return fallback_diagnosis("The mechanic returned a diagnosis that could not be parsed.")
    if not isinstance(parsed, dict):
        return fallback_diagnosis("The mechanic returned a diagnosis that could not be parsed.")

    def _text(field: str, limit: int) -> str:
        value = parsed.get(field)
        return value.strip()[:limit] if isinstance(value, str) and value.strip() else ""

    action = parsed.get("recommended_action")
    if action not in RECOMMENDED_ACTIONS:
        action = "none"
    commit = parsed.get("commit")
    if not isinstance(commit, str) or not _COMMIT_PATTERN.match(commit.strip()):
        commit = None
    else:
        commit = commit.strip()
    # A rollback without a validated target commit is not actionable.
    if action == "rollback_worker" and commit is None:
        action = "none"
    summary = _text("summary", 600) or "The mechanic produced no summary."
    return {
        "summary": summary,
        "likely_cause": _text("likely_cause", 400) or "unknown",
        "recommended_action": action,
        "commit": commit,
        "rationale": _text("rationale", 400)
        or "The mechanic gave no rationale; review the evidence before acting.",
    }


def diagnose_worker(llm: LLM, evidence: dict) -> dict:
    evidence_json = json.dumps(evidence, indent=2, default=str)[:12_000]
    raw = llm.complete(PROMPT_TEMPLATE.format(evidence=evidence_json))
    return parse_diagnosis(raw)
