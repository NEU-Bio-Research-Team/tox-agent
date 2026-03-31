from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from tools import check_model_server_health, validate_smiles

from .adk_compat import LlmAgent, ParallelAgent, SequentialAgent
from .researcher_agent import researcher_agent, run_research
from .screening_agent import run_screening, screening_agent
from .writer_agent import build_final_report, writer_agent

VALIDATOR_MODEL = os.getenv(
    "AGENT_MODEL_FAST",
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
)

_SMILES_TOKEN = re.compile(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+")


def _looks_like_smiles(token: str) -> bool:
    if not token or len(token) < 2:
        return False

    # Ignore plain lowercase words that are unlikely to be chemistry strings.
    if token.isalpha() and token.lower() == token:
        return False

    # For pure alphabetic tokens, avoid natural-language words like "Phan".
    if token.isalpha():
        upper_count = sum(1 for ch in token if ch.isupper())
        if len(token) > 3 and upper_count < 2:
            return False

    # Keep tokens that contain common SMILES symbols or uppercase atom notation.
    has_symbol = any(ch in token for ch in "[]()=#@+-/\\")
    has_upper = any("A" <= ch <= "Z" for ch in token)
    has_digit = any(ch.isdigit() for ch in token)

    return has_symbol or has_upper or has_digit


def extract_smiles_from_text(user_text: str) -> Optional[str]:
    """Best-effort SMILES extraction from free text."""
    if not user_text or not user_text.strip():
        return None

    raw_candidates = _SMILES_TOKEN.findall(user_text)
    candidates = [c for c in dict.fromkeys(raw_candidates) if _looks_like_smiles(c)]
    candidates.sort(key=len, reverse=True)

    for candidate in candidates:
        validation = validate_smiles(candidate)
        if validation.get("valid"):
            return candidate

    return None


def run_input_validation(smiles_input: str) -> Dict[str, Any]:
    """Health-check + SMILES validation stage."""
    health = check_model_server_health()
    validation = validate_smiles(smiles_input)

    if not health.get("healthy"):
        return {
            "validation_status": "INVALID",
            "validation_error": "model_server_unhealthy",
            "health": health,
            "smiles_validation": validation,
            "canonical_smiles": None,
        }

    if not validation.get("valid"):
        return {
            "validation_status": "INVALID",
            "validation_error": validation.get("error") or "invalid_smiles",
            "health": health,
            "smiles_validation": validation,
            "canonical_smiles": None,
        }

    return {
        "validation_status": "VALID",
        "validation_error": None,
        "health": health,
        "smiles_validation": validation,
        "canonical_smiles": validation.get("canonical_smiles") or smiles_input,
    }


def run_orchestrator_flow(smiles_input: str, max_literature_results: int = 5) -> Dict[str, Any]:
    """Deterministic orchestration flow for local and CI smoke tests."""
    state: Dict[str, Any] = {
        "smiles_input": smiles_input,
        "validation_status": "INVALID",
        "screening_result": None,
        "screening_error": None,
        "research_result": None,
        "final_report": None,
    }

    validation = run_input_validation(smiles_input)
    state["validation_status"] = validation["validation_status"]
    state["validation_error"] = validation.get("validation_error")
    state["health"] = validation.get("health")
    state["smiles_validation"] = validation.get("smiles_validation")

    if validation["validation_status"] != "VALID":
        state["final_report"] = {
            "report_metadata": {"smiles": smiles_input, "report_version": "1.0"},
            "error": validation.get("validation_error"),
            "executive_summary": "Validation failed before running parallel agents.",
            "risk_level": "UNKNOWN",
            "sections": {},
        }
        return state

    canonical_smiles = validation["canonical_smiles"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        screening_future = executor.submit(run_screening, canonical_smiles)
        research_future = executor.submit(run_research, canonical_smiles, max_literature_results)

        screening_payload = screening_future.result()
        research_payload = research_future.result()

    state["screening_result"] = screening_payload.get("screening_result")
    state["screening_error"] = screening_payload.get("screening_error")
    state["research_result"] = research_payload.get("research_result")
    state["research_error"] = research_payload.get("research_error")

    state["final_report"] = build_final_report(
        smiles_input=smiles_input,
        screening_result=state["screening_result"],
        research_result=state["research_result"],
    )

    return state


def run_orchestrator_from_text(user_text: str, max_literature_results: int = 5) -> Dict[str, Any]:
    """Parse free text input and execute orchestration flow."""
    smiles = extract_smiles_from_text(user_text)
    if not smiles:
        return {
            "smiles_input": None,
            "validation_status": "INVALID",
            "validation_error": "smiles_not_found",
            "screening_result": None,
            "research_result": None,
            "final_report": {
                "report_metadata": {"smiles": None, "report_version": "1.0"},
                "error": "smiles_not_found",
                "executive_summary": "No valid SMILES could be extracted from input text.",
                "risk_level": "UNKNOWN",
                "sections": {},
            },
        }

    return run_orchestrator_flow(smiles, max_literature_results=max_literature_results)


input_validator = LlmAgent(
    name="InputValidator",
    model=VALIDATOR_MODEL,
    description="Validate model server health and SMILES input before running analysis.",
    instruction="""
You are the pipeline gatekeeper.

Task:
1. Read SMILES from {smiles_input}.
2. Call check_model_server_health().
3. Call validate_smiles(smiles={smiles_input}).
4. Return JSON for key validation_result with fields:
   - validation_status: VALID or INVALID
   - validation_error: null or error string
   - canonical_smiles: canonical string when valid
   - health
   - smiles_validation

Rules:
- If health is unhealthy, mark validation_status as INVALID.
- If SMILES is invalid, mark validation_status as INVALID.
- Never skip tool calls.
""",
    tools=[check_model_server_health, validate_smiles],
    output_key="validation_result",
)

parallel_analysis = ParallelAgent(
    name="ParallelAnalysis",
    sub_agents=[screening_agent, researcher_agent],
    description="Run ScreeningAgent and ResearcherAgent in parallel after validation.",
)

orchestrator = SequentialAgent(
    name="ToxAgentOrchestrator",
    sub_agents=[input_validator, parallel_analysis, writer_agent],
    description="Orchestrate full toxicity pipeline: validate -> parallel analysis -> synthesis.",
)

root_agent = orchestrator
