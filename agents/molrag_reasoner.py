from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

try:
    from google import genai
except Exception:
    genai = None

from .language import choose_text, normalize_language
from services.knowledge_retriever import retrieve_knowledge_context
from services.prompt_builder import build_molrag_prompt

MOLRAG_MODEL = os.getenv("AGENT_MODEL_FAST", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))


def _normalize_label(label: Any) -> str:
    value = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"non_toxic", "nontoxic", "safe", "non-toxic"}:
        return "NON_TOXIC"
    if value in {"toxic", "1"}:
        return "TOXIC"
    return "UNKNOWN"


def _baseline_label_from_prediction(baseline_prediction: Dict[str, Any]) -> str:
    label = str(baseline_prediction.get("label") or "").strip()
    if label:
        return label

    score = baseline_prediction.get("score")
    try:
        return "Toxic" if float(score) >= 0.5 else "Non-toxic"
    except Exception:
        return "UNKNOWN"


def _count_labels(retrieved_examples: List[Dict[str, Any]]) -> Tuple[int, int]:
    toxic_count = 0
    non_toxic_count = 0
    for item in retrieved_examples:
        label = str(item.get("label") or "").strip().lower()
        if "non" in label or "safe" in label:
            non_toxic_count += 1
        elif "toxic" in label or label == "1":
            toxic_count += 1
    return toxic_count, non_toxic_count


def _compose_context_summary(
    *,
    language: str,
    tox_classes: List[str],
    knowledge_hits: List[Dict[str, Any]],
    literature_hits: List[Dict[str, Any]],
) -> str:
    top_knowledge_names = [str(item.get("name") or "").strip() for item in knowledge_hits[:2] if str(item.get("name") or "").strip()]
    top_literature_titles = [str(item.get("title") or "").strip() for item in literature_hits[:2] if str(item.get("title") or "").strip()]
    tox_preview = ", ".join(tox_classes[:3]) if tox_classes else "none"

    return choose_text(
        language,
        (
            f"Knowledge hits={len(knowledge_hits)}, literature hits={len(literature_hits)}, tox_class={tox_preview}. "
            f"Co che noi bat: {', '.join(top_knowledge_names) if top_knowledge_names else 'khong co'}; "
            f"bai bao noi bat: {', '.join(top_literature_titles) if top_literature_titles else 'khong co'}."
        ),
        (
            f"Knowledge hits={len(knowledge_hits)}, literature hits={len(literature_hits)}, tox_class={tox_preview}. "
            f"Top mechanisms: {', '.join(top_knowledge_names) if top_knowledge_names else 'none'}; "
            f"top papers: {', '.join(top_literature_titles) if top_literature_titles else 'none'}."
        ),
    )


def _deterministic_reasoning(
    *,
    input_smiles: str,
    retrieved_examples: List[Dict[str, Any]],
    baseline_prediction: Dict[str, Any],
    knowledge_context: Dict[str, Any],
    language: str,
    strategy: str,
) -> Dict[str, Any]:
    baseline_label = _baseline_label_from_prediction(baseline_prediction)
    baseline_label_normalized = _normalize_label(baseline_label)
    tox_classes = [str(item) for item in knowledge_context.get("tox_classes", [])]
    knowledge_hits = [item for item in knowledge_context.get("knowledge_hits", []) if isinstance(item, dict)]
    literature_hits = [item for item in knowledge_context.get("literature_hits", []) if isinstance(item, dict)]
    toxic_count, non_toxic_count = _count_labels(retrieved_examples)
    top_similarity = max((float(item.get("similarity", 0.0) or 0.0) for item in retrieved_examples), default=0.0)
    high_risk_mechanisms = sum(
        1
        for item in knowledge_hits
        if str(item.get("risk_level") or "").strip().lower() in {"high", "severe"}
    )

    if toxic_count > non_toxic_count:
        suggested_label = "Toxic"
    elif non_toxic_count > toxic_count:
        suggested_label = "Non-toxic"
    else:
        suggested_label = baseline_label

    confidence = round(
        min(
            0.95,
            0.4
            + top_similarity * 0.4
            + min(len(retrieved_examples), 5) * 0.05
            + min(len(knowledge_hits), 4) * 0.015
            + min(len(literature_hits), 4) * 0.01,
        ),
        3,
    )

    analog_evidence = choose_text(
        language,
        (
            f"Tim thay {len(retrieved_examples)} analog, top similarity={top_similarity:.2f}, "
            f"toxic={toxic_count}, non_toxic={non_toxic_count}."
        ),
        (
            f"Retrieved {len(retrieved_examples)} analogs with top similarity={top_similarity:.2f}, "
            f"toxic={toxic_count}, non_toxic={non_toxic_count}."
        ),
    )
    context_evidence = _compose_context_summary(
        language=language,
        tox_classes=tox_classes,
        knowledge_hits=knowledge_hits,
        literature_hits=literature_hits,
    )
    evidence_summary = f"{analog_evidence} {context_evidence}".strip()

    if not retrieved_examples and not knowledge_hits and not literature_hits:
        reasoning_summary = choose_text(
            language,
            "Khong tim thay analog du manh, vi vay MolRAG chi dong vai tro ghi chu bo sung va giu ket qua baseline.",
            "No strong analogs were retrieved, so MolRAG acts as supporting context and keeps the baseline result.",
        )
    elif high_risk_mechanisms > 0 and baseline_label_normalized == "NON_TOXIC":
        reasoning_summary = choose_text(
            language,
            (
                "Bang chung co xuat hien co che nguy co cao trong knowledge base, "
                "vi vay can than trong dien giai du baseline dang non-toxic."
            ),
            (
                "High-risk mechanism signals appeared in the knowledge base, "
                "so the baseline non-toxic conclusion should be interpreted cautiously."
            ),
        )
    elif suggested_label == baseline_label:
        reasoning_summary = choose_text(
            language,
            f"Bang chung tu analog dang dong thuan voi baseline, nen giai thich MolRAG ung ho nhan {baseline_label}.",
            f"The analog evidence is aligned with the baseline, so MolRAG supports the {baseline_label} label.",
        )
    else:
        reasoning_summary = choose_text(
            language,
            f"Bang chung analog co xu huong nghieng ve {suggested_label} nhung MVP van giu baseline lam nguon quyet dinh cuoi.",
            f"The analog evidence leans toward {suggested_label}, but the MVP still keeps the baseline as the final decision source.",
        )

    should_override_with_alignment = bool(retrieved_examples) and not (
        high_risk_mechanisms > 0 and baseline_label_normalized == "NON_TOXIC"
    )
    if should_override_with_alignment:
        suggested_label_normalized = _normalize_label(suggested_label)
        if suggested_label_normalized == baseline_label_normalized and suggested_label_normalized != "UNKNOWN":
            reasoning_summary = choose_text(
                language,
                f"Bang chung tu analog dang dong thuan voi baseline, nen giai thich MolRAG ung ho nhan {baseline_label}.",
                f"The analog evidence is aligned with the baseline, so MolRAG supports the {baseline_label} label.",
            )
        elif suggested_label_normalized != "UNKNOWN" and baseline_label_normalized != "UNKNOWN":
            reasoning_summary = choose_text(
                language,
                f"Bang chung analog co xu huong nghieng ve {suggested_label} nhung MVP van giu baseline lam nguon quyet dinh cuoi.",
                f"The analog evidence leans toward {suggested_label}, but the MVP still keeps the baseline as the final decision source.",
            )

    return {
        "enabled": True,
        "strategy": strategy,
        "input_smiles": input_smiles,
        "reasoning_mode": "deterministic",
        "evidence_summary": evidence_summary,
        "reasoning_summary": reasoning_summary,
        "suggested_label": suggested_label,
        "confidence": confidence,
        "tox_classes": tox_classes,
        "knowledge_hits": knowledge_hits,
        "literature_hits": literature_hits,
        "knowledge_error": knowledge_context.get("error"),
        "firestore": knowledge_context.get("firestore"),
    }


def run_molrag_reasoning(
    *,
    input_smiles: str,
    retrieved_examples: List[Dict[str, Any]],
    baseline_prediction: Dict[str, Any],
    language: str = "vi",
    strategy: str = "sim_cot",
) -> Dict[str, Any]:
    normalized_language = normalize_language(language)
    knowledge_context = retrieve_knowledge_context(
        input_smiles=input_smiles,
        retrieved_examples=retrieved_examples,
    )
    knowledge_hits = [item for item in knowledge_context.get("knowledge_hits", []) if isinstance(item, dict)]
    literature_hits = [item for item in knowledge_context.get("literature_hits", []) if isinstance(item, dict)]

    prompt = build_molrag_prompt(
        input_smiles=input_smiles,
        language=normalized_language,
        baseline_prediction=baseline_prediction,
        retrieved_examples=retrieved_examples,
        knowledge_hits=knowledge_hits,
        literature_hits=literature_hits,
        strategy=strategy,
    )

    result = _deterministic_reasoning(
        input_smiles=input_smiles,
        retrieved_examples=retrieved_examples,
        baseline_prediction=baseline_prediction,
        knowledge_context=knowledge_context,
        language=normalized_language,
        strategy=strategy,
    )
    result["prompt_preview"] = prompt[:1200]

    if genai is None or not os.getenv("MOLRAG_ENABLE_LLM"):
        return result

    # Keep the prototype resilient: deterministic reasoning remains the default.
    result["llm_status"] = "llm_not_invoked_in_prototype"
    return result
