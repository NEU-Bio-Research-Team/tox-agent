from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

try:
    from google import genai
except Exception:
    genai = None

from .language import choose_text, normalize_language
from services.prompt_builder import build_molrag_prompt

MOLRAG_MODEL = os.getenv("AGENT_MODEL_FAST", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))


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


def _deterministic_reasoning(
    *,
    input_smiles: str,
    retrieved_examples: List[Dict[str, Any]],
    baseline_prediction: Dict[str, Any],
    language: str,
    strategy: str,
) -> Dict[str, Any]:
    baseline_label = _baseline_label_from_prediction(baseline_prediction)
    toxic_count, non_toxic_count = _count_labels(retrieved_examples)
    top_similarity = max((float(item.get("similarity", 0.0) or 0.0) for item in retrieved_examples), default=0.0)

    if toxic_count > non_toxic_count:
        suggested_label = "Toxic"
    elif non_toxic_count > toxic_count:
        suggested_label = "Non-toxic"
    else:
        suggested_label = baseline_label

    confidence = round(min(0.95, 0.4 + top_similarity * 0.4 + min(len(retrieved_examples), 5) * 0.05), 3)

    evidence_summary = choose_text(
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

    if not retrieved_examples:
        reasoning_summary = choose_text(
            language,
            "Khong tim thay analog du manh, vi vay MolRAG chi dong vai tro ghi chu bo sung va giu ket qua baseline.",
            "No strong analogs were retrieved, so MolRAG acts as supporting context and keeps the baseline result.",
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

    return {
        "enabled": True,
        "strategy": strategy,
        "input_smiles": input_smiles,
        "reasoning_mode": "deterministic",
        "evidence_summary": evidence_summary,
        "reasoning_summary": reasoning_summary,
        "suggested_label": suggested_label,
        "confidence": confidence,
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
    prompt = build_molrag_prompt(
        input_smiles=input_smiles,
        language=normalized_language,
        baseline_prediction=baseline_prediction,
        retrieved_examples=retrieved_examples,
        strategy=strategy,
    )

    result = _deterministic_reasoning(
        input_smiles=input_smiles,
        retrieved_examples=retrieved_examples,
        baseline_prediction=baseline_prediction,
        language=normalized_language,
        strategy=strategy,
    )
    result["prompt_preview"] = prompt[:1200]

    if genai is None or not os.getenv("MOLRAG_ENABLE_LLM"):
        return result

    # Keep the prototype resilient: deterministic reasoning remains the default.
    result["llm_status"] = "llm_not_invoked_in_prototype"
    return result
