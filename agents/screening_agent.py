from __future__ import annotations

import os
from typing import Any, Dict

from tools import analyze_molecule, validate_smiles

from .adk_compat import LlmAgent
from .language import choose_text
from .molrag_reasoner import run_molrag_reasoning
from services import fuse_molrag_with_baseline, retrieve_similar_molecules

SCREENING_MODEL = os.getenv("AGENT_MODEL_FAST", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))


def run_screening(
    smiles_input: str,
    language: str = "vi",
    clinical_threshold: float = 0.35,
    mechanism_threshold: float = 0.5,
    inference_backend: str = "xsmiles",
    binary_tox_model: str = "pretrained_2head_herg_chemberta_model",
    tox_type_model: str = "tox21_ensemble_3_best",
    molrag_enabled: bool = False,
    molrag_top_k: int = 5,
    molrag_min_similarity: float = 0.15,
) -> Dict[str, Any]:
    """Deterministic screening flow used for local tests and orchestration."""
    try:
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
            inference_backend=inference_backend,
            binary_tox_model=binary_tox_model,
            tox_type_model=tox_type_model,
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
        baseline_prediction = {
            "label": clinical.get("label"),
            "score": clinical.get("p_toxic"),
            "confidence": clinical.get("confidence"),
            "ood_flag": bool(ood_assessment.get("flag", False)),
        }

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

        molrag_payload = {
            "enabled": False,
            "strategy": "sim_cot",
            "retrieved_examples": [],
            "reasoning_summary": None,
            "suggested_label": None,
            "confidence": None,
            "error": None,
        }
        fusion_result = None

        if molrag_enabled:
            retrieval_payload = retrieve_similar_molecules(
                canonical_smiles,
                top_k=molrag_top_k,
                min_similarity=molrag_min_similarity,
            )
            retrieved_examples = retrieval_payload.get("matches", [])
            molrag_reasoning = run_molrag_reasoning(
                input_smiles=canonical_smiles,
                retrieved_examples=retrieved_examples,
                baseline_prediction=baseline_prediction,
                language=language,
            )
            molrag_payload = {
                "enabled": True,
                "strategy": molrag_reasoning.get("strategy", "sim_cot"),
                "retrieval_db_size": retrieval_payload.get("db_size"),
                "retrieval_db_source": retrieval_payload.get("db_source"),
                "retrieval_error": retrieval_payload.get("error"),
                "firestore": retrieval_payload.get("firestore") or molrag_reasoning.get("firestore"),
                "retrieved_examples": retrieved_examples,
                "evidence_summary": molrag_reasoning.get("evidence_summary"),
                "reasoning_summary": molrag_reasoning.get("reasoning_summary"),
                "suggested_label": molrag_reasoning.get("suggested_label"),
                "confidence": molrag_reasoning.get("confidence"),
                "tox_classes": molrag_reasoning.get("tox_classes", []),
                "knowledge_hits": molrag_reasoning.get("knowledge_hits", []),
                "literature_hits": molrag_reasoning.get("literature_hits", []),
                "knowledge_error": molrag_reasoning.get("knowledge_error"),
                "prompt_preview": molrag_reasoning.get("prompt_preview"),
                "reasoning_mode": molrag_reasoning.get("reasoning_mode"),
                "error": retrieval_payload.get("error") or molrag_reasoning.get("error"),
            }
            fusion_result = fuse_molrag_with_baseline(
                baseline_prediction=baseline_prediction,
                molrag_result=molrag_reasoning,
                mode="evidence_only",
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
            "baseline_prediction": baseline_prediction,
            "molrag": molrag_payload,
            "fusion_result": fusion_result,
            "final_verdict": final_verdict,
            "error": None,
        }

        return {
            "screening_result": screening_result,
            "screening_error": None,
            "canonical_smiles": screening_result.get("canonical_smiles"),
        }
    except Exception as exc:
        return {
            "screening_result": None,
            "screening_error": f"run_screening_exception: {type(exc).__name__}: {str(exc)[:200]}",
            "canonical_smiles": None,
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
1. Read canonical SMILES from session state key {canonical_smiles}.
2. If {canonical_smiles} is missing/empty, fall back to {smiles_input}.
2. Read language from {language} and write user-facing text in that language.
3. Read thresholds from {clinical_threshold} and {mechanism_threshold}.
4. Read inference backend from {inference_backend}.
5. Read binary model key from {binary_tox_model}.
6. Read tox-type model key from {tox_type_model}.
5. Do NOT call validate_smiles. Input is already validated by InputValidator.
6. Call analyze_molecule(
    smiles=<canonical_smiles from session state>,
    clinical_threshold={clinical_threshold},
    mechanism_threshold={mechanism_threshold},
    inference_backend={inference_backend},
    binary_tox_model={binary_tox_model},
    tox_type_model={tox_type_model}
).
6. Return JSON for key screening_result with fields:
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
    tools=[analyze_molecule],
    output_key="screening_result",
)
