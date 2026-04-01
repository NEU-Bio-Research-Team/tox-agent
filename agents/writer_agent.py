from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .adk_compat import LlmAgent
from .language import choose_text, normalize_language

try:
    from google import genai
except Exception:
    genai = None

WRITER_MODEL = os.getenv("AGENT_MODEL_PRO", "gemini-2.5-pro")
FALSE_NEGATIVE_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent / "test_data" / "false_negative_registry.json"
)


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


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_llm_recommendations(raw_text: str) -> List[str]:
    candidate = _strip_code_fence(raw_text)
    payload: Any = None

    try:
        payload = json.loads(candidate)
    except Exception:
        match = re.search(r"\{[\s\S]*\}", candidate)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = None

    if not isinstance(payload, dict):
        # Fallback parser when the model returns plain bullets instead of JSON.
        lines = [line.strip() for line in candidate.splitlines() if line.strip()]
        parsed_lines: List[str] = []
        for line in lines:
            cleaned = re.sub(r"^[\-\*\d\.)\s]+", "", line).strip()
            if cleaned:
                parsed_lines.append(cleaned)
        return parsed_lines[:5]

    recs = payload.get("recommendations")
    if not isinstance(recs, list):
        return []

    cleaned: List[str] = []
    for rec in recs:
        text = str(rec or "").strip()
        if text:
            cleaned.append(text)
    return cleaned[:5]


def _default_recommendations(
    risk_level: str,
    mechanism: Dict[str, Any],
    language: str,
    clinical: Dict[str, Any],
    ood_assessment: Dict[str, Any],
    inference_context: Dict[str, Any],
) -> List[str]:
    highest_risk_task = mechanism.get("highest_risk_task")
    assay_hits = int(mechanism.get("assay_hits", 0) or 0)
    p_toxic = float(clinical.get("p_toxic", 0.0) or 0.0)
    threshold_used = float(clinical.get("threshold_used", 0.35) or 0.35)
    threshold_policy = str(inference_context.get("threshold_policy") or "balanced").upper()
    ood_flag = bool(ood_assessment.get("flag", False))
    ood_risk = str(ood_assessment.get("ood_risk") or "LOW")
    rare_elements = ood_assessment.get("high_risk_elements") or ood_assessment.get("rare_elements") or []
    if not isinstance(rare_elements, list):
        rare_elements = []

    recs: List[str] = [
        choose_text(
            language,
            (
                f"Bối cảnh quyết định hiện tại: p_toxic={p_toxic:.3f}, "
                f"ngưỡng={threshold_used:.2f}, assay_hits={assay_hits}."
            ),
            (
                f"Current decision context: p_toxic={p_toxic:.3f}, "
                f"threshold={threshold_used:.2f}, assay_hits={assay_hits}."
            ),
        )
    ]

    recs.append(
        choose_text(
            language,
            f"Threshold policy đang áp dụng: {threshold_policy}.",
            f"Current threshold policy in use: {threshold_policy}.",
        )
    )

    if risk_level == "CRITICAL":
        recs.extend(choose_text(
            language,
            [
                "Ưu tiên xác nhận in-vitro ngay và mở review dược hóa chuyên sâu.",
                "Tạm dừng progression cho đến khi có kế hoạch giảm thiểu theo cơ chế độc tính.",
            ],
            [
                "Escalate to immediate in-vitro follow-up and medicinal chemistry review.",
                "Block progression until mechanism-specific safety mitigation is defined.",
            ],
        ))
    elif risk_level == "HIGH":
        recs.extend(choose_text(
            language,
            [
                "Chạy assay xác nhận tập trung vào các cơ chế có điểm rủi ro cao.",
                "Cân nhắc tối ưu scaffold/substituent tại các vùng đóng góp độc tính cao.",
            ],
            [
                "Run targeted confirmatory assays for predicted mechanism liabilities.",
                "Consider structure edits around high-contributing substructures.",
            ],
        ))
    elif risk_level == "MODERATE":
        recs.extend(choose_text(
            language,
            [
                "Ưu tiên assay orthogonal cho pathway rủi ro cao nhất.",
                "Theo dõi độ bất định và so sánh với các analog gần cấu trúc.",
            ],
            [
                "Prioritize orthogonal assays for the top risk pathway.",
                "Track uncertainty and compare with close structural analogs.",
            ],
        ))
    else:
        recs.extend(choose_text(
            language,
            [
                "Tiếp tục panel xác nhận tiêu chuẩn với kiểm soát an toàn mặc định.",
                "Đánh giá lại rủi ro sau mọi thay đổi scaffold hoặc substituent.",
            ],
            [
                "Proceed with routine validation panel under standard safety controls.",
                "Re-check risk after any scaffold or substituent modifications.",
            ],
        ))

    if highest_risk_task:
        recs.append(
            choose_text(
                language,
                f"Assay ưu tiên vòng tiếp theo: {highest_risk_task}.",
                f"Focus next mechanism assay on {highest_risk_task}.",
            )
        )

    if ood_flag:
        element_text = ", ".join(str(item) for item in rare_elements) if rare_elements else "N/A"
        recs.append(
            choose_text(
                language,
                f"Gắn cờ OOD mức {ood_risk}. Nguyên tố hiếm: {element_text}.",
                f"OOD is flagged at level {ood_risk}. Rare elements: {element_text}.",
            )
        )

    return recs


def _build_llm_prompt(
    language: str,
    risk_level: str,
    clinical: Dict[str, Any],
    mechanism: Dict[str, Any],
    ood_assessment: Dict[str, Any],
    research: Dict[str, Any],
) -> str:
    language_name = "Vietnamese" if language == "vi" else "English"
    prompt_payload = {
        "risk_level": risk_level,
        "clinical": {
            "label": clinical.get("label"),
            "p_toxic": clinical.get("p_toxic"),
            "threshold_used": clinical.get("threshold_used"),
            "confidence": clinical.get("confidence"),
        },
        "mechanism": {
            "assay_hits": mechanism.get("assay_hits"),
            "highest_risk_task": mechanism.get("highest_risk_task"),
            "highest_risk_score": mechanism.get("highest_risk_score"),
            "active_tasks": mechanism.get("active_tasks"),
        },
        "ood_assessment": {
            "ood_risk": ood_assessment.get("ood_risk"),
            "flag": ood_assessment.get("flag"),
            "reason": ood_assessment.get("reason"),
            "recommendation": ood_assessment.get("recommendation"),
        },
        "literature": {
            "query_name_used": research.get("query_name_used"),
            "total_found": (research.get("literature") or {}).get("total_found"),
            "tox21_active_count": (research.get("bioassay_summary") or {}).get("tox21_active_count"),
        },
    }

    return (
        "You are a medicinal safety advisor drafting action items for a toxicity screening report. "
        f"Output language must be {language_name}.\n"
        "Return STRICT JSON only with schema:\n"
        "{\"recommendations\": [\"...\", \"...\", \"...\"]}\n"
        "Rules:\n"
        "- Provide 3 to 5 concise action items.\n"
        "- Mention threshold policy and OOD handling when relevant.\n"
        "- Do not output markdown or prose outside JSON.\n"
        f"Input: {json.dumps(prompt_payload, ensure_ascii=True)}"
    )


def _maybe_llm_recommendations(
    language: str,
    risk_level: str,
    clinical: Dict[str, Any],
    mechanism: Dict[str, Any],
    ood_assessment: Dict[str, Any],
    research: Dict[str, Any],
) -> Tuple[List[str], str]:
    enabled = str(os.getenv("WRITER_ENABLE_LLM_RECOMMENDATIONS", "1")).strip().lower()
    if enabled in {"0", "false", "no"}:
        return [], "llm_disabled_by_env"
    if genai is None:
        return [], "google_genai_not_available"

    try:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return [], "missing_gemini_api_key"

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=WRITER_MODEL,
            contents=_build_llm_prompt(
                language=language,
                risk_level=risk_level,
                clinical=clinical,
                mechanism=mechanism,
                ood_assessment=ood_assessment,
                research=research,
            ),
            config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            },
        )
        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            return [], "llm_empty_response"

        parsed = _parse_llm_recommendations(text)
        if not parsed:
            return [], "llm_parse_failed"

        return parsed, "llm_success"
    except Exception as exc:
        return [], f"llm_error:{type(exc).__name__}"


def _build_recommendations(
    risk_level: str,
    mechanism: Dict[str, Any],
    language: str,
    clinical: Dict[str, Any],
    ood_assessment: Dict[str, Any],
    research: Dict[str, Any],
    inference_context: Dict[str, Any],
) -> Tuple[List[str], str, str]:
    dynamic_recs, detail = _maybe_llm_recommendations(
        language=language,
        risk_level=risk_level,
        clinical=clinical,
        mechanism=mechanism,
        ood_assessment=ood_assessment,
        research=research,
    )
    if dynamic_recs:
        return dynamic_recs, "llm", detail

    return (
        _default_recommendations(
            risk_level=risk_level,
            mechanism=mechanism,
            language=language,
            clinical=clinical,
            ood_assessment=ood_assessment,
            inference_context=inference_context,
        ),
        "deterministic",
        detail,
    )


@lru_cache(maxsize=1)
def _load_false_negative_registry() -> List[Dict[str, Any]]:
    if not FALSE_NEGATIVE_REGISTRY_PATH.exists():
        return []
    try:
        with open(FALSE_NEGATIVE_REGISTRY_PATH, "r") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    except Exception:
        return []
    return []


def _lookup_failure_registry(canonical_smiles: str | None) -> Tuple[Optional[Dict[str, Any]], int]:
    registry = _load_false_negative_registry()
    if not canonical_smiles:
        return None, len(registry)

    normalized = str(canonical_smiles).strip()
    for row in registry:
        if str(row.get("canonical_smiles", "")).strip() == normalized:
            return row, len(registry)

    return None, len(registry)


def build_final_report(
    smiles_input: str,
    screening_result: Dict[str, Any] | None,
    research_result: Dict[str, Any] | None,
    language: str = "vi",
) -> Dict[str, Any]:
    """Build the final structured report from screening and research outputs."""
    screening = _to_dict(screening_result)
    research = _to_dict(research_result)
    normalized_language = normalize_language(language or screening.get("language") or research.get("language"))

    if not screening:
        return {
            "report_metadata": {
                "smiles": smiles_input,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "report_version": "1.0",
            },
            "error": "screening_result_missing",
            "executive_summary": choose_text(
                normalized_language,
                "Không thể tạo báo cáo vì bước screening bị lỗi.",
                "Unable to generate report because screening stage failed.",
            ),
            "risk_level": "UNKNOWN",
            "sections": {},
        }

    clinical = _to_dict(screening.get("clinical"))
    mechanism = _to_dict(screening.get("mechanism"))
    explanation = _to_dict(screening.get("explanation"))
    ood_assessment = _to_dict(screening.get("ood_assessment"))
    inference_context = _to_dict(screening.get("inference_context"))
    reliability_warning = screening.get("reliability_warning")

    compound_info = _to_dict(research.get("compound_info"))
    literature = _to_dict(research.get("literature"))
    bioassay_summary = research.get("bioassay_summary")

    risk_level = _compute_risk_level(clinical, mechanism)
    recommendations, recommendation_source, recommendation_source_detail = _build_recommendations(
        risk_level=risk_level,
        mechanism=mechanism,
        language=normalized_language,
        clinical=clinical,
        ood_assessment=ood_assessment,
        research=research,
        inference_context=inference_context,
    )

    failure_match, registry_size = _lookup_failure_registry(
        str(screening.get("canonical_smiles") or "").strip() or None
    )

    if failure_match is not None:
        recommendations.insert(
            0,
            choose_text(
                normalized_language,
                f"Khớp Failure Registry ({failure_match.get('id', 'N/A')}): ưu tiên expert review.",
                f"Failure Registry match ({failure_match.get('id', 'N/A')}): route to expert review.",
            ),
        )

    compound_name = compound_info.get("common_name") or compound_info.get("iupac_name")

    executive_summary = choose_text(
        normalized_language,
        (
            f"Phân tử được phân loại {clinical.get('label', 'N/A')} với mức rủi ro {risk_level}. "
            f"Final verdict: {screening.get('final_verdict', 'UNKNOWN')}."
        ),
        (
            f"Molecule is classified as {clinical.get('label', 'N/A')} with "
            f"risk level {risk_level}. Final verdict: {screening.get('final_verdict', 'UNKNOWN')}."
        ),
    )

    bioassay_explanation = choose_text(
        normalized_language,
        "Bioassay data là kết quả thí nghiệm sinh học công khai (PubChem) cho hợp chất này, dùng để đối chiếu tín hiệu cơ chế từ model với bằng chứng thực nghiệm đã được báo cáo.",
        "Bioassay data are public biological assay outcomes (PubChem) for this compound, used to cross-check model mechanism signals against prior experimental evidence.",
    )

    return {
        "report_metadata": {
            "smiles": smiles_input,
            "canonical_smiles": screening.get("canonical_smiles"),
            "compound_name": compound_name,
            "language": normalized_language,
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
                "threshold_used": clinical.get("threshold_used"),
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
                "molecule_png_base64": explanation.get("molecule_png_base64"),
                "target_task": explanation.get("target_task"),
                "target_task_score": explanation.get("target_task_score"),
                "explainer_note": explanation.get("explainer_note"),
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
                "bioassay_explanation": bioassay_explanation,
            },
            "ood_assessment": ood_assessment,
            "inference_context": inference_context,
            "reliability_warning": reliability_warning,
            "recommendation_source": recommendation_source,
            "recommendation_source_detail": recommendation_source_detail,
            "failure_registry": {
                "matched": failure_match is not None,
                "entry": failure_match,
                "registry_size": registry_size,
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
- language

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
- Respect requested language from {language} for all user-facing text.
- If research_result is partial/missing, still produce a valid report.
""",
    tools=[],
    output_key="final_report",
)
