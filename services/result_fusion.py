from __future__ import annotations

from typing import Any, Dict


def _normalize_label(label: Any) -> str:
    value = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"toxic", "1", "positive"}:
        return "TOXIC"
    if value in {"non_toxic", "nontoxic", "safe", "non-toxic", "0", "negative"}:
        return "NON_TOXIC"
    return "UNKNOWN"


def fuse_molrag_with_baseline(
    *,
    baseline_prediction: Dict[str, Any],
    molrag_result: Dict[str, Any],
    mode: str = "evidence_only",
) -> Dict[str, Any]:
    baseline_label = str(baseline_prediction.get("label") or "UNKNOWN")
    baseline_score = baseline_prediction.get("score")
    baseline_confidence = baseline_prediction.get("confidence")
    molrag_label = str(molrag_result.get("suggested_label") or "UNKNOWN")
    molrag_confidence = molrag_result.get("confidence")
    baseline_label_normalized = _normalize_label(baseline_label)
    molrag_label_normalized = _normalize_label(molrag_label)

    return {
        "mode": mode,
        "baseline_label": baseline_label,
        "baseline_score": baseline_score,
        "baseline_confidence": baseline_confidence,
        "molrag_label": molrag_label,
        "molrag_confidence": molrag_confidence,
        "final_label": baseline_label,
        "final_confidence": baseline_confidence,
        "agreement": (
            baseline_label_normalized == molrag_label_normalized
            if molrag_label_normalized != "UNKNOWN"
            else None
        ),
        "decision_note": (
            "Baseline model remains the source of truth in MVP mode."
            if mode == "evidence_only"
            else "Fusion mode is not enabled in this prototype."
        ),
    }
