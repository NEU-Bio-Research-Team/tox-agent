from __future__ import annotations

import os
from typing import Any, Dict

from tools import analyze_molecule, validate_smiles

from .adk_compat import LlmAgent

SCREENING_MODEL = os.getenv("AGENT_MODEL_FAST", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))


def run_screening(smiles_input: str) -> Dict[str, Any]:
    """Deterministic screening flow used for local tests and orchestration."""
    validation = validate_smiles(smiles_input)
    if not validation.get("valid"):
        return {
            "screening_result": None,
            "screening_error": validation.get("error") or "invalid_smiles",
            "canonical_smiles": None,
        }

    canonical_smiles = validation.get("canonical_smiles") or smiles_input.strip()
    analysis = analyze_molecule(canonical_smiles)

    if analysis.get("error"):
        return {
            "screening_result": None,
            "screening_error": analysis.get("error"),
            "canonical_smiles": canonical_smiles,
            "analysis_raw": analysis,
        }

    clinical = analysis.get("clinical", {})
    mechanism = analysis.get("mechanism", {})
    explanation = analysis.get("explanation", {})
    final_verdict = analysis.get("final_verdict", "UNKNOWN")

    summary = (
        f"Clinical={clinical.get('label', 'N/A')} "
        f"(p_toxic={clinical.get('p_toxic', 'N/A')}), "
        f"assay_hits={mechanism.get('assay_hits', 0)}, "
        f"verdict={final_verdict}."
    )

    screening_result = {
        "summary": summary,
        "smiles": analysis.get("smiles", smiles_input),
        "canonical_smiles": analysis.get("canonical_smiles", canonical_smiles),
        "clinical": clinical,
        "mechanism": mechanism,
        "explanation": explanation,
        "final_verdict": final_verdict,
        "error": None,
    }

    return {
        "screening_result": screening_result,
        "screening_error": None,
        "canonical_smiles": screening_result.get("canonical_smiles"),
    }


screening_agent = LlmAgent(
    name="ScreeningAgent",
    model=SCREENING_MODEL,
    description=(
        "Analyze clinical and mechanistic toxicity from a SMILES string."
    ),
    instruction="""
You are a molecular toxicity screening specialist.

Task:
1. Read SMILES from session state key {smiles_input}.
2. Call validate_smiles(smiles={smiles_input}).
3. If valid, call analyze_molecule(smiles=<canonical_smiles from validate step>).
4. Return JSON for key screening_result with fields:
   - summary
   - smiles
   - canonical_smiles
   - clinical
   - mechanism
   - explanation
   - final_verdict
   - error

Rules:
- Never fabricate prediction values.
- Always use tool outputs as source of truth.
- If any tool fails, set error and keep remaining fields as null/empty.
""",
    tools=[validate_smiles, analyze_molecule],
    output_key="screening_result",
)
