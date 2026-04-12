from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from tools import validate_smiles

from .adk_compat import LlmAgent, ParallelAgent, SequentialAgent
from .language import choose_text, normalize_language
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
    """SMILES validation stage.

    Health checks are intentionally excluded from gating here because
    internal self-calls can produce false negatives under load.
    """
    validation = validate_smiles(smiles_input)

    if not validation.get("valid"):
        return {
            "validation_status": "INVALID",
            "validation_error": validation.get("error") or "invalid_smiles",
            "health": {
                "healthy": True,
                "status": "skipped",
                "note": "health_check_skipped_internal_validation",
                "error": None,
            },
            "smiles_validation": validation,
            "canonical_smiles": None,
        }

    return {
        "validation_status": "VALID",
        "validation_error": None,
        "health": {
            "healthy": True,
            "status": "skipped",
            "note": "health_check_skipped_internal_validation",
            "error": None,
        },
        "smiles_validation": validation,
        "canonical_smiles": validation.get("canonical_smiles") or smiles_input,
    }


def run_orchestrator_flow(
    smiles_input: str,
    max_literature_results: int = 5,
    language: str = "vi",
    clinical_threshold: float = 0.35,
    mechanism_threshold: float = 0.5,
    inference_backend: str = "xsmiles",
    binary_tox_model: str = "pretrained_2head_herg_chemberta_model",
    tox_type_model: str = "tox21_ensemble_3_best",
) -> Dict[str, Any]:
    """Deterministic orchestration flow for local and CI smoke tests."""
    normalized_language = normalize_language(language)
    state: Dict[str, Any] = {
        "smiles_input": smiles_input,
        "language": normalized_language,
        "clinical_threshold": float(clinical_threshold),
        "mechanism_threshold": float(mechanism_threshold),
        "inference_backend": str(inference_backend),
        "binary_tox_model": str(binary_tox_model),
        "tox_type_model": str(tox_type_model),
        "validation_status": "INVALID",
        "screening_result": None,
        "explanation_raw": None,
        "screening_error": None,
        "research_result": None,
        "research_error": None,
        "final_report": None,
    }

    validation = run_input_validation(smiles_input)
    state["validation_status"] = validation["validation_status"]
    state["validation_error"] = validation.get("validation_error")
    state["health"] = validation.get("health")
    state["smiles_validation"] = validation.get("smiles_validation")

    if validation["validation_status"] != "VALID":
        state["final_report"] = {
            "report_metadata": {
                "smiles": smiles_input,
                "report_version": "1.0",
                "language": normalized_language,
            },
            "error": validation.get("validation_error"),
            "executive_summary": choose_text(
                normalized_language,
                "SMILES không hợp lệ, nên pipeline dừng ở bước validation.",
                "SMILES validation failed before running parallel agents.",
            ),
            "risk_level": "UNKNOWN",
            "sections": {
                "clinical_toxicity": {},
                "mechanism_toxicity": {},
            },
        }
        return state

    canonical_smiles = validation.get("canonical_smiles") or smiles_input

    with ThreadPoolExecutor(max_workers=2) as executor:
        screening_future = executor.submit(
            run_screening,
            canonical_smiles,
            normalized_language,
            float(clinical_threshold),
            float(mechanism_threshold),
            str(inference_backend),
            str(binary_tox_model),
            str(tox_type_model),
        )
        research_future = executor.submit(
            run_research,
            canonical_smiles,
            max_literature_results,
            normalized_language,
        )

        screening_payload = screening_future.result()
        research_payload = research_future.result()

    state["screening_result"] = screening_payload.get("screening_result")
    if isinstance(state.get("screening_result"), dict):
        state["explanation_raw"] = state["screening_result"].get("explanation")
    state["screening_error"] = screening_payload.get("screening_error")
    state["research_result"] = research_payload.get("research_result")
    state["research_error"] = research_payload.get("research_error")

    state["final_report"] = build_final_report(
        smiles_input=smiles_input,
        screening_result=state["screening_result"],
        research_result=state["research_result"],
        explanation_raw=state.get("explanation_raw"),
        language=normalized_language,
    )

    return state


def run_orchestrator_from_text(
    user_text: str,
    max_literature_results: int = 5,
    language: str = "vi",
    clinical_threshold: float = 0.35,
    mechanism_threshold: float = 0.5,
    inference_backend: str = "xsmiles",
    binary_tox_model: str = "pretrained_2head_herg_chemberta_model",
    tox_type_model: str = "tox21_ensemble_3_best",
) -> Dict[str, Any]:
    """Parse free text input and execute orchestration flow."""

    normalized_language = normalize_language(language)
    smiles = extract_smiles_from_text(user_text)
    if not smiles:
        return {
            "smiles_input": None,
            "language": normalized_language,
            "validation_status": "INVALID",
            "validation_error": "smiles_not_found",
            "screening_result": None,
            "research_result": None,
            "final_report": {
                "report_metadata": {
                    "smiles": None,
                    "report_version": "1.0",
                    "language": normalized_language,
                },
                "error": "smiles_not_found",
                "executive_summary": choose_text(
                    normalized_language,
                    "Không trích xuất được SMILES hợp lệ từ nội dung đầu vào.",
                    "No valid SMILES could be extracted from input text.",
                ),
                "risk_level": "UNKNOWN",
                "sections": {},
            },
        }

    return run_orchestrator_flow(
        smiles,
        max_literature_results=max_literature_results,
        language=normalized_language,
        clinical_threshold=clinical_threshold,
        mechanism_threshold=mechanism_threshold,
        inference_backend=inference_backend,
        binary_tox_model=binary_tox_model,
        tox_type_model=tox_type_model,
    )


def _build_input_validator_instruction() -> str:
    return """
You are the pipeline gatekeeper.

Task:
1. Read SMILES from {smiles_input}.
2. Read language from {language} (vi or en) and use it for all user-facing text.
3. Call validate_smiles(smiles={smiles_input}).
4. Return STRICT RAW JSON (no markdown code fences) for key validation_result with EXACT schema:
{
    "validation_result": {
        "validation_status": "VALID" | "INVALID",
        "validation_error": null | string,
        "canonical_smiles": string | null,
        "health": {
            "healthy": true,
            "status": "skipped",
            "note": "health_check_skipped_internal_validation",
            "error": null
        },
        "smiles_validation": object
    }
}

Rules:
- Do not call check_model_server_health for validation gating.
- If SMILES is invalid, set validation_status=INVALID and validation_error from tool output.
- If SMILES is valid, set validation_status=VALID and canonical_smiles from tool output.
- Return raw JSON only.
"""


input_validator = LlmAgent(
    name="InputValidator",
    model=VALIDATOR_MODEL,
    description="Validate SMILES input before running analysis.",
    instruction=_build_input_validator_instruction(),
    tools=[validate_smiles],
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
