from __future__ import annotations

import os
from typing import Any, Dict

from tools import analyze_molecule, validate_smiles

from .adk_compat import LlmAgent
from .language import choose_text

SCREENING_MODEL = os.getenv("AGENT_MODEL_FAST", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))


def run_screening(
    smiles_input: str,
    language: str = "vi",
    clinical_threshold: float = 0.35,
    mechanism_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Deterministic screening flow used for local tests and orchestration."""
    validation = validate_smiles(smiles_input)
    if not validation.get("valid"):
        return {
            "screening_result": None,
            "screening_error": validation.get("error") or "invalid_smiles",
            "canonical_smiles": None,
        }

    canonical_smiles = validation.get("canonical_smiles") or smiles_input.strip()
    analysis = analyze_molecule(
        canonical_smiles,
        clinical_threshold=clinical_threshold,
        mechanism_threshold=mechanism_threshold,
    )

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
    ood_assessment = analysis.get("ood_assessment", {})
    threshold_used = clinical.get("threshold_used")
    ood_risk = ood_assessment.get("ood_risk", "LOW")

    summary = choose_text(
        language,
        (
            f"Lâm sàng={clinical.get('label', 'N/A')} "
            f"(p_toxic={clinical.get('p_toxic', 'N/A')}, ngưỡng={threshold_used}), "
            f"assay_hits={mechanism.get('assay_hits', 0)}, kết luận={final_verdict}, "
            f"OOD={ood_risk}."
        ),
        (
            f"Clinical={clinical.get('label', 'N/A')} "
            f"(p_toxic={clinical.get('p_toxic', 'N/A')}, threshold={threshold_used}), "
            f"assay_hits={mechanism.get('assay_hits', 0)}, verdict={final_verdict}, "
            f"OOD={ood_risk}."
        ),
    )

    screening_result = {
        "summary": summary,
        "smiles": analysis.get("smiles", smiles_input),
        "canonical_smiles": analysis.get("canonical_smiles", canonical_smiles),
        "clinical": clinical,
        "mechanism": mechanism,
        "explanation": explanation,
        "ood_assessment": ood_assessment,
        "reliability_warning": analysis.get("reliability_warning"),
        "inference_context": analysis.get("inference_context"),
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
2. Read language from {language} and write user-facing text in that language.
3. Read thresholds from {clinical_threshold} and {mechanism_threshold}.
4. Call validate_smiles(smiles={smiles_input}).
4. If valid, call analyze_molecule(
    smiles=<canonical_smiles from validate step>,
    clinical_threshold={clinical_threshold},
    mechanism_threshold={mechanism_threshold}
).
5. Return JSON for key screening_result with fields:
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
