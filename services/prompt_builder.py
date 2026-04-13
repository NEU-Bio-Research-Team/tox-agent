from __future__ import annotations

import json
from typing import Any, Dict, List

from agents.language import choose_text


def build_molrag_prompt(
    *,
    input_smiles: str,
    language: str,
    baseline_prediction: Dict[str, Any],
    retrieved_examples: List[Dict[str, Any]],
    strategy: str = "sim_cot",
) -> str:
    task_instruction = choose_text(
        language,
        "Hay su dung cac phan tu tuong tu de giai thich ket qua du doan doc tinh hien tai.",
        "Use the retrieved analog molecules to explain the current toxicity prediction.",
    )

    payload = {
        "strategy": strategy,
        "input_smiles": input_smiles,
        "baseline_prediction": baseline_prediction,
        "retrieved_examples": retrieved_examples,
    }

    return (
        f"{task_instruction}\n"
        "Return structured reasoning with: evidence_summary, reasoning_summary, suggested_label, confidence.\n"
        f"Context JSON: {json.dumps(payload, ensure_ascii=True)}"
    )
