"""Deterministic activity presentation registry.

The agent never invents a progress statement: a label is selected from the
tool actually being executed and may safely be localized by the client.
"""
from __future__ import annotations

from typing import Any

_TOOLS: dict[str, tuple[str, str, str]] = {
    "search_toxicology_evidence": ("retrieval", "literature_search", "activity.searching_literature"),
    "get_evidence_record": ("reading", "evidence_reading", "activity.reading_sources"),
    "predict_toxicity": ("prediction", "toxicity_prediction", "activity.running_predictor"),
    "recognize_structure": ("prediction", "structure_recognition", "activity.recognizing_structure"),
    "get_analysis_slice": ("analysis", "cross_check", "activity.reviewing_results"),
    "submit_grounded_answer": ("synthesis", "answer_synthesis", "activity.synthesizing"),
}


def activity_for_tool(tool_name: str, *, status: str, progress: dict[str, int] | None = None) -> dict[str, Any]:
    phase, kind, label_key = _TOOLS.get(
        tool_name, ("analysis", "cross_check", "activity.processing")
    )
    payload: dict[str, Any] = {
        "phase": phase,
        "kind": kind,
        "status": status,
        "label_key": label_key,
        "visibility": "user",
    }
    if progress:
        payload["progress"] = progress
    return payload
