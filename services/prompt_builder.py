from __future__ import annotations

import json
from typing import Any, Dict, List


def _normalize_language(language: str | None) -> str:
    value = str(language or "vi").strip().lower()
    return "en" if value.startswith("en") else "vi"


def _choose_text(language: str, vi_text: str, en_text: str) -> str:
    return en_text if _normalize_language(language) == "en" else vi_text


def build_molrag_prompt(
    *,
    input_smiles: str,
    language: str,
    baseline_prediction: Dict[str, Any],
    retrieved_examples: List[Dict[str, Any]],
    knowledge_hits: List[Dict[str, Any]] | None = None,
    literature_hits: List[Dict[str, Any]] | None = None,
    strategy: str = "sim_cot",
) -> str:
    task_instruction = _choose_text(
        language,
        "Hay su dung cac phan tu tuong tu de giai thich ket qua du doan doc tinh hien tai.",
        "Use the retrieved analog molecules to explain the current toxicity prediction.",
    )

    payload = {
        "strategy": strategy,
        "input_smiles": input_smiles,
        "baseline_prediction": baseline_prediction,
        "retrieved_examples": retrieved_examples,
        "knowledge_hits": knowledge_hits or [],
        "literature_hits": literature_hits or [],
    }

    return (
        f"{task_instruction}\n"
        "Return structured reasoning with: evidence_summary, reasoning_summary, suggested_label, confidence.\n"
        f"Context JSON: {json.dumps(payload, ensure_ascii=True)}"
    )
