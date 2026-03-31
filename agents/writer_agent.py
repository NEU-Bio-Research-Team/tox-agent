from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from .adk_compat import LlmAgent

WRITER_MODEL = os.getenv("AGENT_MODEL_PRO", "gemini-2.5-pro")


def _to_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _compute_risk_level(clinical: Dict[str, Any], mechanism: Dict[str, Any]) -> str:
    p_toxic = float(clinical.get("p_toxic", 0.0) or 0.0)

    active_tasks = mechanism.get("active_tasks")
    if isinstance(active_tasks, list):
        active_count = len(active_tasks)
    else:
        active_count = int(mechanism.get("assay_hits", 0) or 0)

    if p_toxic > 0.8 and active_count >= 3:
        return "CRITICAL"
    if p_toxic > 0.6 or active_count >= 2:
        return "HIGH"
    if p_toxic > 0.4 or active_count >= 1:
        return "MODERATE"
    return "LOW"


def _build_recommendations(risk_level: str, mechanism: Dict[str, Any]) -> List[str]:
    highest_risk_task = mechanism.get("highest_risk_task")

    if risk_level == "CRITICAL":
        recs = [
            "Escalate to immediate in-vitro follow-up and medicinal chemistry review.",
            "Block progression until mechanism-specific safety mitigation is defined.",
        ]
    elif risk_level == "HIGH":
        recs = [
            "Run targeted confirmatory assays for predicted mechanism liabilities.",
            "Consider structure edits around high-contributing substructures.",
        ]
    elif risk_level == "MODERATE":
        recs = [
            "Prioritize orthogonal assays for the top risk pathway.",
            "Track uncertainty and compare with close structural analogs.",
        ]
    else:
        recs = [
            "Proceed with routine validation panel under standard safety controls.",
            "Re-check risk after any scaffold or substituent modifications.",
        ]

    if highest_risk_task:
        recs.append(f"Focus next mechanism assay on {highest_risk_task}.")

    return recs


def build_final_report(
    smiles_input: str,
    screening_result: Dict[str, Any] | None,
    research_result: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Build the final structured report from screening and research outputs."""
    screening = _to_dict(screening_result)
    research = _to_dict(research_result)

    if not screening:
        return {
            "report_metadata": {
                "smiles": smiles_input,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "report_version": "1.0",
            },
            "error": "screening_result_missing",
            "executive_summary": "Unable to generate report because screening stage failed.",
            "risk_level": "UNKNOWN",
            "sections": {},
        }

    clinical = _to_dict(screening.get("clinical"))
    mechanism = _to_dict(screening.get("mechanism"))
    explanation = _to_dict(screening.get("explanation"))

    compound_info = _to_dict(research.get("compound_info"))
    literature = _to_dict(research.get("literature"))
    bioassay_summary = research.get("bioassay_summary")

    risk_level = _compute_risk_level(clinical, mechanism)
    recommendations = _build_recommendations(risk_level, mechanism)

    compound_name = compound_info.get("common_name") or compound_info.get("iupac_name")

    executive_summary = (
        f"Molecule is classified as {clinical.get('label', 'N/A')} with "
        f"risk level {risk_level}. Final verdict: {screening.get('final_verdict', 'UNKNOWN')}."
    )

    return {
        "report_metadata": {
            "smiles": smiles_input,
            "canonical_smiles": screening.get("canonical_smiles"),
            "compound_name": compound_name,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "report_version": "1.0",
        },
        "executive_summary": executive_summary,
        "risk_level": risk_level,
        "sections": {
            "clinical_toxicity": {
                "verdict": clinical.get("label"),
                "probability": clinical.get("p_toxic"),
                "confidence": clinical.get("confidence"),
                "interpretation": screening.get("summary"),
            },
            "mechanism_toxicity": {
                "active_tox21_tasks": mechanism.get("active_tasks", []),
                "highest_risk": mechanism.get("highest_risk_task"),
                "assay_hits": mechanism.get("assay_hits"),
                "task_scores": mechanism.get("task_scores"),
            },
            "structural_explanation": {
                "top_atoms": explanation.get("top_atoms"),
                "top_bonds": explanation.get("top_bonds"),
                "heatmap_base64": explanation.get("heatmap_base64"),
            },
            "literature_context": {
                "compound_id": {
                    "cid": compound_info.get("cid"),
                    "pubchem_url": compound_info.get("pubchem_url"),
                },
                "query_name_used": research.get("query_name_used"),
                "total_found": literature.get("total_found"),
                "relevant_papers": literature.get("articles", []),
                "bioassay_evidence": bioassay_summary,
            },
            "recommendations": recommendations,
        },
    }


writer_agent = LlmAgent(
    name="WriterAgent",
    model=WRITER_MODEL,
    description="Synthesize screening and research outputs into a structured final report.",
        instruction="""
You are a toxicity report writer.

Read from session state:
- smiles_input
- screening_result
- research_result

Output requirements:
- Return JSON for key final_report.
- Include:
    report_metadata, executive_summary, risk_level, sections.
- Sections must include:
    clinical_toxicity, mechanism_toxicity, structural_explanation,
    literature_context, recommendations.

Risk policy:
- CRITICAL: p_toxic > 0.8 AND assay_hits >= 3
- HIGH: p_toxic > 0.6 OR assay_hits >= 2
- MODERATE: p_toxic > 0.4 OR assay_hits >= 1
- LOW: otherwise

Rules:
- Never fabricate scores or papers.
- If research_result is partial/missing, still produce a valid report.
""",
    tools=[],
    output_key="final_report",
)
