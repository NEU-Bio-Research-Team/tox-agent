from __future__ import annotations

from typing import Any, Dict


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

    return {
        "mode": mode,
        "baseline_label": baseline_label,
        "baseline_score": baseline_score,
        "baseline_confidence": baseline_confidence,
        "molrag_label": molrag_label,
        "molrag_confidence": molrag_confidence,
        "final_label": baseline_label,
        "final_confidence": baseline_confidence,
        "agreement": baseline_label == molrag_label if molrag_label != "UNKNOWN" else None,
        "decision_note": (
            "Baseline model remains the source of truth in MVP mode."
            if mode == "evidence_only"
            else "Fusion mode is not enabled in this prototype."
        ),
    }
