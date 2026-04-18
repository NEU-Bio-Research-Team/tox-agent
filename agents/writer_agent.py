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


def _to_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _to_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(parsed, 0)


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


def _normalize_priority(value: Any) -> str:
    raw = str(value or "MEDIUM").strip().upper()
    if raw in {"HIGH", "MEDIUM", "LOW"}:
        return raw
    return "MEDIUM"


def _normalize_action_type(value: Any) -> str:
    raw = str(value or "monitoring").strip().lower()
    if raw in {"experimental", "structural", "regulatory", "monitoring"}:
        return raw
    return "monitoring"


def _to_recommendation_item(
    *,
    action: str,
    rationale: str,
    priority: str = "MEDIUM",
    action_type: str = "monitoring",
) -> Dict[str, str]:
    return {
        "priority": _normalize_priority(priority),
        "action_type": _normalize_action_type(action_type),
        "action": str(action or "").strip(),
        "rationale": str(rationale or "").strip(),
    }


def _parse_llm_recommendations(raw_text: str) -> List[Dict[str, str]]:
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
        parsed_lines: List[Dict[str, str]] = []
        for line in lines:
            cleaned = re.sub(r"^[\-\*\d\.)\s]+", "", line).strip()
            if cleaned:
                parsed_lines.append(
                    _to_recommendation_item(
                        action=cleaned,
                        rationale="Generated from free-text fallback response.",
                        priority="MEDIUM",
                        action_type="monitoring",
                    )
                )
        return parsed_lines[:5]

    recs = payload.get("recommendations")
    if not isinstance(recs, list):
        return []

    cleaned: List[Dict[str, str]] = []
    for rec in recs:
        if isinstance(rec, str):
            text = rec.strip()
            if text:
                cleaned.append(
                    _to_recommendation_item(
                        action=text,
                        rationale="Model returned unstructured recommendation text.",
                        priority="MEDIUM",
                        action_type="monitoring",
                    )
                )
            continue

        if not isinstance(rec, dict):
            continue

        action = str(rec.get("action") or "").strip()
        rationale = str(rec.get("rationale") or "").strip()
        if not action:
            continue

        cleaned.append(
            _to_recommendation_item(
                action=action,
                rationale=rationale,
                priority=rec.get("priority", "MEDIUM"),
                action_type=rec.get("action_type", "monitoring"),
            )
        )
    return cleaned[:5]


def _default_recommendations(
    risk_level: str,
    mechanism: Dict[str, Any],
    language: str,
    clinical: Dict[str, Any],
    ood_assessment: Dict[str, Any],
    inference_context: Dict[str, Any],
) -> List[Dict[str, str]]:
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

    recs: List[Dict[str, str]] = [
        _to_recommendation_item(
            priority="MEDIUM",
            action_type="monitoring",
            action=choose_text(
                language,
                "Theo dõi lại quyết định độc tính theo policy ngưỡng hiện tại.",
                "Re-check the toxicity decision under the active threshold policy.",
            ),
            rationale=choose_text(
                language,
                (
                    f"Bối cảnh hiện tại: p_toxic={p_toxic:.3f}, ngưỡng={threshold_used:.2f}, "
                    f"assay_hits={assay_hits}, threshold_policy={threshold_policy}."
                ),
                (
                    f"Current context: p_toxic={p_toxic:.3f}, threshold={threshold_used:.2f}, "
                    f"assay_hits={assay_hits}, threshold_policy={threshold_policy}."
                ),
            ),
        )
    ]

    if ood_flag:
        element_text = ", ".join(str(item) for item in rare_elements) if rare_elements else "N/A"
        recs.insert(
            0,
            _to_recommendation_item(
                priority="HIGH",
                action_type="regulatory",
                action=choose_text(
                    language,
                    "Khoá quyết định progression và yêu cầu review chuyên gia do cờ OOD.",
                    "Gate progression and require expert adjudication due to OOD flag.",
                ),
                rationale=choose_text(
                    language,
                    f"OOD={ood_risk}; nguyên tố hiếm={element_text}.",
                    f"OOD={ood_risk}; rare elements={element_text}.",
                ),
            ),
        )

    if risk_level == "CRITICAL":
        recs.extend(
            [
                _to_recommendation_item(
                    priority="HIGH",
                    action_type="experimental",
                    action=choose_text(
                        language,
                        "Ưu tiên assay xác nhận in-vitro khẩn cấp cho các tín hiệu độc tính chính.",
                        "Run immediate in-vitro confirmatory assays for top toxicity signals.",
                    ),
                    rationale=choose_text(
                        language,
                        f"Risk level={risk_level} với assay_hits={assay_hits} và p_toxic={p_toxic:.3f}.",
                        f"Risk level={risk_level} with assay_hits={assay_hits} and p_toxic={p_toxic:.3f}.",
                    ),
                ),
                _to_recommendation_item(
                    priority="HIGH",
                    action_type="structural",
                    action=choose_text(
                        language,
                        "Tạm dừng progression và mở vòng tối ưu cấu trúc để giảm liability chính.",
                        "Pause progression and launch structural optimization against primary liabilities.",
                    ),
                    rationale=choose_text(
                        language,
                        "Mức rủi ro CRITICAL yêu cầu giảm thiểu trước khi quyết định tiếp theo.",
                        "CRITICAL risk requires mitigation before next-stage decisions.",
                    ),
                ),
            ]
        )
    elif risk_level == "HIGH":
        recs.extend(
            [
                _to_recommendation_item(
                    priority="HIGH",
                    action_type="experimental",
                    action=choose_text(
                        language,
                        "Chạy panel assay xác nhận tập trung vào pathway rủi ro cao.",
                        "Run targeted confirmatory assay panel on highest-risk pathways.",
                    ),
                    rationale=choose_text(
                        language,
                        f"Mức HIGH với assay_hits={assay_hits} và p_toxic={p_toxic:.3f}.",
                        f"HIGH risk with assay_hits={assay_hits} and p_toxic={p_toxic:.3f}.",
                    ),
                ),
                _to_recommendation_item(
                    priority="MEDIUM",
                    action_type="structural",
                    action=choose_text(
                        language,
                        "Đề xuất tối ưu scaffold/substituent quanh vùng đóng góp độc tính cao.",
                        "Propose scaffold/substituent optimization around high-risk contribution regions.",
                    ),
                    rationale=choose_text(
                        language,
                        "Giảm liability cấu trúc có thể hạ xác suất toxic trong vòng lặp kế tiếp.",
                        "Structural liability reduction can lower toxic probability in next iteration.",
                    ),
                ),
            ]
        )
    elif risk_level == "MODERATE":
        recs.extend(
            [
                _to_recommendation_item(
                    priority="MEDIUM",
                    action_type="experimental",
                    action=choose_text(
                        language,
                        "Ưu tiên assay orthogonal cho cơ chế có điểm cao nhất.",
                        "Prioritize orthogonal assays for the highest-scoring mechanism.",
                    ),
                    rationale=choose_text(
                        language,
                        "Mức MODERATE cần xác nhận thực nghiệm để giảm bất định.",
                        "MODERATE risk requires experimental confirmation to reduce uncertainty.",
                    ),
                ),
                _to_recommendation_item(
                    priority="MEDIUM",
                    action_type="monitoring",
                    action=choose_text(
                        language,
                        "So sánh profile với các analog gần cấu trúc trước khi nâng cấp quyết định.",
                        "Compare profile with close analogs before escalating decisions.",
                    ),
                    rationale=choose_text(
                        language,
                        "Đối chiếu analog giúp kiểm tra tính ổn định tín hiệu độc tính.",
                        "Analog comparison helps verify signal stability.",
                    ),
                ),
            ]
        )
    else:
        recs.extend(
            [
                _to_recommendation_item(
                    priority="LOW",
                    action_type="experimental",
                    action=choose_text(
                        language,
                        "Duy trì panel xác nhận tiêu chuẩn theo quy trình safety hiện tại.",
                        "Continue routine validation panel under current safety workflow.",
                    ),
                    rationale=choose_text(
                        language,
                        f"Risk={risk_level} với p_toxic={p_toxic:.3f} dưới ngưỡng cảnh báo cao.",
                        f"Risk={risk_level} with p_toxic={p_toxic:.3f} below high-alert territory.",
                    ),
                ),
                _to_recommendation_item(
                    priority="LOW",
                    action_type="monitoring",
                    action=choose_text(
                        language,
                        "Đánh giá lại rủi ro sau mỗi thay đổi scaffold/substituent.",
                        "Re-evaluate risk after each scaffold/substituent change.",
                    ),
                    rationale=choose_text(
                        language,
                        "Ngay cả profile LOW cũng cần theo dõi khi tối ưu hóa lead.",
                        "Even LOW-risk profiles should be tracked during lead optimization.",
                    ),
                ),
            ]
        )

    if highest_risk_task:
        recs.append(
            _to_recommendation_item(
                priority="MEDIUM",
                action_type="experimental",
                action=choose_text(
                    language,
                    f"Ưu tiên assay vòng kế tiếp: {highest_risk_task}.",
                    f"Prioritize next assay cycle on {highest_risk_task}.",
                ),
                rationale=choose_text(
                    language,
                    "Assay này là tín hiệu cơ chế nổi bật nhất trong lần screening hiện tại.",
                    "This assay is the strongest mechanism signal in the current screening output.",
                ),
            )
        )

    return recs[:5]


def _build_llm_prompt(
    language: str,
    risk_level: str,
    smiles: str,
    compound_name: Optional[str],
    clinical: Dict[str, Any],
    mechanism: Dict[str, Any],
    ood_assessment: Dict[str, Any],
    research: Dict[str, Any],
    molrag: Dict[str, Any],
    fusion_result: Dict[str, Any],
) -> str:
    language_name = "Vietnamese" if language == "vi" else "English"

    task_scores = mechanism.get("task_scores") if isinstance(mechanism.get("task_scores"), dict) else {}
    active_tasks = mechanism.get("active_tasks") if isinstance(mechanism.get("active_tasks"), list) else []
    ranked_tasks = sorted(
        [
            (str(task), float(task_scores.get(task, 0.0) or 0.0))
            for task in active_tasks
            if str(task).strip()
        ],
        key=lambda item: item[1],
        reverse=True,
    )[:4]

    article_titles: List[str] = []
    literature_articles = _to_dict(research.get("literature")).get("articles")
    if isinstance(literature_articles, list):
        for article in literature_articles[:4]:
            if not isinstance(article, dict):
                continue
            title = str(article.get("title") or "").strip()
            if title:
                article_titles.append(title)

    prompt_payload = {
        "compound": {
            "smiles": smiles,
            "name": compound_name or "Unknown",
        },
        "risk_level": risk_level,
        "clinical_prediction": {
            "label": clinical.get("label"),
            "p_toxic": clinical.get("p_toxic"),
            "threshold_used": clinical.get("threshold_used"),
            "confidence": clinical.get("confidence"),
        },
        "mechanism_signals": {
            "assay_hits": mechanism.get("assay_hits"),
            "highest_risk_task": mechanism.get("highest_risk_task"),
            "highest_risk_score": mechanism.get("highest_risk_score"),
            "active_tasks": mechanism.get("active_tasks"),
            "top_active_tasks_with_scores": ranked_tasks,
        },
        "molrag_evidence": {
            "label": molrag.get("suggested_label") or molrag.get("label"),
            "confidence": molrag.get("confidence"),
            "tox_classes": (molrag.get("tox_classes") or [])[:5],
            "analogs_found": len(molrag.get("retrieved_examples") or []),
        },
        "model_fusion": {
            "agreement": fusion_result.get("agreement"),
            "final_label": fusion_result.get("final_label"),
            "final_confidence": fusion_result.get("final_confidence"),
        },
        "ood_assessment": {
            "ood_risk": ood_assessment.get("ood_risk"),
            "flag": ood_assessment.get("flag"),
            "reason": ood_assessment.get("reason"),
            "recommendation": ood_assessment.get("recommendation"),
        },
        "literature_signals": {
            "query_name_used": research.get("query_name_used"),
            "total_found": (research.get("literature") or {}).get("total_found"),
            "tox21_active_count": (research.get("bioassay_summary") or {}).get("tox21_active_count"),
            "representative_titles": article_titles,
        },
    }

    output_schema = {
        "recommendations": [
            {
                "priority": "HIGH|MEDIUM|LOW",
                "action_type": "experimental|structural|regulatory|monitoring",
                "action": "<clear actionable instruction>",
                "rationale": "<one sentence grounded in input data>",
            }
        ]
    }

    return (
        "You are a senior medicinal safety advisor drafting action items for a toxicity screening report. "
        f"Output language must be {language_name}.\n"
        "Return STRICT JSON only with schema:\n"
        f"{json.dumps(output_schema, ensure_ascii=True)}\n"
        "Rules:\n"
        "- Provide 3 to 5 recommendations.\n"
        "- Each recommendation must include priority, action_type, action, and rationale.\n"
        "- Ground each rationale in actual input signals (scores, tasks, OOD, MolRAG, literature).\n"
        "- Cover at least one experimental and one monitoring action type where possible.\n"
        "- If fusion agreement is false, include an expert adjudication recommendation.\n"
        "- Do not output markdown or prose outside JSON.\n"
        f"Input: {json.dumps(prompt_payload, ensure_ascii=False)}"
    )


def _build_genai_client(location_override: Optional[str] = None) -> Tuple[Optional[Any], str]:
    """Create a genai client using API key if present, otherwise Vertex AI ADC."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            return genai.Client(api_key=api_key), "api_key"
        except Exception as exc:
            return None, f"api_key_client_error:{type(exc).__name__}"

    project = (
        os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
    )
    if not project:
        return None, "missing_project_for_vertexai"

    configured_location = (
        location_override
        or os.getenv("GEMINI_LOCATION")
        or os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("GOOGLE_CLOUD_REGION")
        or "global"
    )
    try:
        return (
            genai.Client(
                vertexai=True,
                project=project,
                location=configured_location,
            ),
            f"vertex_adc:{configured_location}",
        )
    except Exception as exc:
        return None, f"vertex_client_error:{type(exc).__name__}"


def _generate_llm_recommendations_with_client(
    *,
    client: Any,
    language: str,
    risk_level: str,
    smiles: str,
    compound_name: Optional[str],
    clinical: Dict[str, Any],
    mechanism: Dict[str, Any],
    ood_assessment: Dict[str, Any],
    research: Dict[str, Any],
    molrag: Dict[str, Any],
    fusion_result: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], str]:
    response = client.models.generate_content(
        model=WRITER_MODEL,
        contents=_build_llm_prompt(
            language=language,
            risk_level=risk_level,
            smiles=smiles,
            compound_name=compound_name,
            clinical=clinical,
            mechanism=mechanism,
            ood_assessment=ood_assessment,
            research=research,
            molrag=molrag,
            fusion_result=fusion_result,
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


def _maybe_llm_recommendations(
    language: str,
    risk_level: str,
    smiles: str,
    compound_name: Optional[str],
    clinical: Dict[str, Any],
    mechanism: Dict[str, Any],
    ood_assessment: Dict[str, Any],
    research: Dict[str, Any],
    molrag: Dict[str, Any],
    fusion_result: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], str]:
    enabled = str(os.getenv("WRITER_ENABLE_LLM_RECOMMENDATIONS", "1")).strip().lower()
    if enabled in {"0", "false", "no"}:
        return [], "llm_disabled_by_env"
    if genai is None:
        return [], "google_genai_not_available"

    client, auth_mode = _build_genai_client()
    if client is None:
        return [], auth_mode

    try:
        parsed, status = _generate_llm_recommendations_with_client(
            client=client,
            language=language,
            risk_level=risk_level,
            smiles=smiles,
            compound_name=compound_name,
            clinical=clinical,
            mechanism=mechanism,
            ood_assessment=ood_assessment,
            research=research,
            molrag=molrag,
            fusion_result=fusion_result,
        )
        if parsed:
            return parsed, f"{status}:{auth_mode}"
        return [], status
    except Exception as exc:
        first_error = f"llm_error:{type(exc).__name__}:{auth_mode}:{str(exc)[:180]}"

        if auth_mode.startswith("vertex_adc:"):
            for fallback_location in ("global", "us-central1"):
                if auth_mode.endswith(fallback_location):
                    continue
                retry_client, retry_auth = _build_genai_client(location_override=fallback_location)
                if retry_client is None:
                    continue
                try:
                    parsed, status = _generate_llm_recommendations_with_client(
                        client=retry_client,
                        language=language,
                        risk_level=risk_level,
                        smiles=smiles,
                        compound_name=compound_name,
                        clinical=clinical,
                        mechanism=mechanism,
                        ood_assessment=ood_assessment,
                        research=research,
                        molrag=molrag,
                        fusion_result=fusion_result,
                    )
                    if parsed:
                        return parsed, f"{status}:{retry_auth}"
                except Exception:
                    continue

        return [], first_error


def _build_recommendations(
    risk_level: str,
    smiles: str,
    compound_name: Optional[str],
    mechanism: Dict[str, Any],
    language: str,
    clinical: Dict[str, Any],
    ood_assessment: Dict[str, Any],
    research: Dict[str, Any],
    molrag: Dict[str, Any],
    fusion_result: Dict[str, Any],
    inference_context: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], str, str]:
    dynamic_recs, detail = _maybe_llm_recommendations(
        language=language,
        risk_level=risk_level,
        smiles=smiles,
        compound_name=compound_name,
        clinical=clinical,
        mechanism=mechanism,
        ood_assessment=ood_assessment,
        research=research,
        molrag=molrag,
        fusion_result=fusion_result,
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
    explanation_raw: Dict[str, Any] | None = None,
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
                "language": normalized_language,
            },
            "error": "screening_result_missing",
            "executive_summary": choose_text(
                normalized_language,
                "Không thể tạo báo cáo vì bước screening bị lỗi.",
                "Unable to generate report because screening stage failed.",
            ),
            "risk_level": "UNKNOWN",
            "sections": {
                "clinical_toxicity": {
                    "verdict": None,
                    "probability": None,
                    "confidence": None,
                    "threshold_used": None,
                    "interpretation": choose_text(
                        normalized_language,
                        "Thiếu dữ liệu screening.",
                        "Screening data is missing.",
                    ),
                },
                "mechanism_toxicity": {
                    "active_tox21_tasks": [],
                    "highest_risk": None,
                    "assay_hits": 0,
                    "task_scores": {},
                },
            },
        }

    clinical = _to_dict(screening.get("clinical"))
    mechanism = _to_dict(screening.get("mechanism"))
    raw_explanation = _to_dict(explanation_raw)
    if isinstance(raw_explanation.get("explanation"), dict):
        raw_explanation = _to_dict(raw_explanation.get("explanation"))
    explanation = raw_explanation or _to_dict(screening.get("explanation"))
    if isinstance(explanation.get("data"), dict):
        explanation = _to_dict(explanation.get("data"))
    ood_assessment = _to_dict(screening.get("ood_assessment"))
    inference_context = _to_dict(screening.get("inference_context"))
    reliability_warning = screening.get("reliability_warning")

    top_atoms = explanation.get("top_atoms")
    if not isinstance(top_atoms, list):
        top_atoms = []

    top_bonds = explanation.get("top_bonds")
    if not isinstance(top_bonds, list):
        top_bonds = []

    heatmap_base64 = explanation.get("heatmap_base64")
    if not isinstance(heatmap_base64, str) or not heatmap_base64.strip():
        heatmap_base64 = None

    molecule_png_base64 = explanation.get("molecule_png_base64")
    if not isinstance(molecule_png_base64, str) or not molecule_png_base64.strip():
        molecule_png_base64 = None

    target_task = explanation.get("target_task") or mechanism.get("highest_risk_task")
    target_task_score = explanation.get("target_task_score")
    if not isinstance(target_task_score, (int, float)):
        target_task_score = mechanism.get("highest_risk_score")

    explainer_note = explanation.get("explainer_note")
    if not isinstance(explainer_note, str) or not explainer_note.strip():
        explainer_note = choose_text(
            normalized_language,
            "Attribution chi tiet chua san sang cho mau nay; payload cau truc duoc giu de on dinh giao dien.",
            "Detailed structural attribution is unavailable for this sample; structural payload is preserved for UI stability.",
        )

    compound_info = _to_dict(research.get("compound_info"))
    literature = _to_dict(research.get("literature"))
    bioassay_summary = research.get("bioassay_summary")
    compound_name = compound_info.get("common_name") or compound_info.get("iupac_name")
    molrag_data = _to_dict(screening.get("molrag"))
    fusion_data = _to_dict(screening.get("fusion_result"))

    literature_articles = _to_list(literature.get("articles"))
    relevant_papers = [article for article in literature_articles if isinstance(article, dict)]

    mechanism_active_tasks = _to_list(mechanism.get("active_tasks"))
    mechanism_active_tasks = [task for task in mechanism_active_tasks if isinstance(task, str)]

    mechanism_task_scores = mechanism.get("task_scores")
    if not isinstance(mechanism_task_scores, dict):
        mechanism_task_scores = {}

    risk_level = _compute_risk_level(clinical, mechanism)
    recommendations, recommendation_source, recommendation_source_detail = _build_recommendations(
        risk_level=risk_level,
        smiles=smiles_input,
        compound_name=compound_name,
        mechanism=mechanism,
        language=normalized_language,
        clinical=clinical,
        ood_assessment=ood_assessment,
        research=research,
        molrag=molrag_data,
        fusion_result=fusion_data,
        inference_context=inference_context,
    )

    failure_match, registry_size = _lookup_failure_registry(
        str(screening.get("canonical_smiles") or "").strip() or None
    )

    if failure_match is not None:
        recommendations.insert(
            0,
            _to_recommendation_item(
                priority="HIGH",
                action_type="regulatory",
                action=choose_text(
                    normalized_language,
                    f"Khớp Failure Registry ({failure_match.get('id', 'N/A')}): ưu tiên expert review.",
                    f"Failure Registry match ({failure_match.get('id', 'N/A')}): route to expert review.",
                ),
                rationale=choose_text(
                    normalized_language,
                    "Mẫu đã ghi nhận tiền lệ false-negative nên cần kiểm định thủ công trước khi ra quyết định.",
                    "The pattern has prior false-negative precedent, so manual adjudication is required before decisions.",
                ),
            ),
        )

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

    clinical_interpretation = screening.get("summary")
    if not isinstance(clinical_interpretation, str) or not clinical_interpretation.strip():
        label = str(clinical.get("label") or "UNKNOWN")
        p_toxic = float(clinical.get("p_toxic", 0.0) or 0.0)
        threshold_used = float(clinical.get("threshold_used", 0.35) or 0.35)
        clinical_interpretation = choose_text(
            normalized_language,
            f"Dien giai lam sang: {label} (p_toxic={p_toxic:.3f}, nguong={threshold_used:.2f}).",
            f"Clinical interpretation: {label} (p_toxic={p_toxic:.3f}, threshold={threshold_used:.2f}).",
        )

    ood_assessment_output = dict(ood_assessment)
    if not isinstance(ood_assessment_output.get("recommendation"), str) or not str(
        ood_assessment_output.get("recommendation")
    ).strip():
        if bool(ood_assessment_output.get("flag", False)):
            ood_assessment_output["recommendation"] = choose_text(
                normalized_language,
                "Co canh bao OOD, nen uu tien xac minh bo sung truoc khi dua ra quyet dinh progression.",
                "OOD is flagged; prioritize additional validation before progression decisions.",
            )
        else:
            ood_assessment_output["recommendation"] = choose_text(
                normalized_language,
                "Khong can hanh dong bo sung cho OOD o lan danh gia nay.",
                "No additional OOD action is required for this evaluation.",
            )

    bioassay_evidence_output = _to_dict(bioassay_summary)
    if not bioassay_evidence_output:
        bioassay_evidence_output = {
            "cid": compound_info.get("cid"),
            "total_assays_tested": 0,
            "active_assays": [],
            "tox21_active_count": 0,
            "error": "not_available",
        }
    else:
        bioassay_evidence_output.setdefault("cid", compound_info.get("cid"))

        active_assays = _to_list(bioassay_evidence_output.get("active_assays"))
        bioassay_evidence_output["active_assays"] = [
            assay for assay in active_assays if isinstance(assay, dict)
        ]

        bioassay_evidence_output["total_assays_tested"] = _to_non_negative_int(
            bioassay_evidence_output.get("total_assays_tested", 0)
        )
        bioassay_evidence_output["tox21_active_count"] = _to_non_negative_int(
            bioassay_evidence_output.get("tox21_active_count", 0)
        )

        err_value = bioassay_evidence_output.get("error")
        if err_value is None or (isinstance(err_value, str) and not err_value.strip()):
            bioassay_evidence_output["error"] = "none"

    failure_registry_entry = (
        failure_match
        if failure_match is not None
        else {
            "status": "not_matched",
            "reason": "no_registry_match",
        }
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
                "interpretation": clinical_interpretation,
            },
            "mechanism_toxicity": {
                "active_tox21_tasks": mechanism_active_tasks,
                "highest_risk": mechanism.get("highest_risk_task"),
                "assay_hits": mechanism.get("assay_hits"),
                "task_scores": mechanism_task_scores,
            },
            "structural_explanation": {
                "top_atoms": top_atoms,
                "top_bonds": top_bonds,
                "heatmap_base64": heatmap_base64,
                "molecule_png_base64": molecule_png_base64,
                "target_task": target_task,
                "target_task_score": target_task_score,
                "explainer_note": explainer_note,
            },
            "molrag_evidence": _to_dict(screening.get("molrag")),
            "fusion_result": _to_dict(screening.get("fusion_result")),
            "literature_context": {
                "compound_id": {
                    "cid": compound_info.get("cid"),
                    "pubchem_url": compound_info.get("pubchem_url"),
                },
                "query_name_used": research.get("query_name_used"),
                "total_found": literature.get("total_found"),
                "relevant_papers": relevant_papers,
                "bioassay_evidence": bioassay_evidence_output,
                "bioassay_explanation": bioassay_explanation,
            },
            "ood_assessment": ood_assessment_output,
            "inference_context": inference_context,
            "reliability_warning": reliability_warning,
            "recommendation_source": recommendation_source,
            "recommendation_source_detail": recommendation_source_detail,
            "failure_registry": {
                "matched": failure_match is not None,
                "entry": failure_registry_entry,
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
- Return STRICT RAW JSON only (no markdown code fences, no prose).
- Return JSON for key final_report.
- Include:
    report_metadata, executive_summary, risk_level, sections.
- sections MUST be a non-empty object.
- Sections must include at minimum:
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
- If screening_result is missing/partial, still return final_report with non-empty
    sections and a clear error in final_report.error.
""",
    tools=[],
    output_key="final_report",
)
