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
    retrieval_context: Dict[str, Any] | None = None,
    firestore_state: Dict[str, Any] | None = None,
    strategy: str = "sim_cot",
) -> str:
    task_instruction = _choose_text(
        language,
        "Hay dong vai tro la lop bang chung MolRAG trung tam, dung analog, knowledge va literature de giai thich sau va ro rang ket qua doc tinh hien tai.",
        "Act as the core MolRAG evidence layer and use analogs, curated knowledge, and literature to deeply explain the current toxicity prediction.",
    )

    payload = {
        "strategy": strategy,
        "input_smiles": input_smiles,
        "baseline_prediction": baseline_prediction,
        "retrieved_examples": retrieved_examples,
        "knowledge_hits": knowledge_hits or [],
        "literature_hits": literature_hits or [],
        "retrieval_context": retrieval_context or {},
        "firestore_state": firestore_state or {},
    }

    return (
        f"{task_instruction}\n"
        "Return structured JSON reasoning with these fields:\n"
        "  evidence_overview: string summarizing retrieval breadth and evidence source quality\n"
        "  longform_summary: detailed 4-6 sentence narrative that reads like an evidence agent report\n"
        "  mechanism_chain: array of reasoning steps (e.g. SMARTS match -> mechanistic liability -> endpoint)\n"
        "  key_substructures: array of structural motifs or substructures driving the call\n"
        "  analogy_reasoning: detailed explanation of how the best analogs support or contradict the verdict\n"
        "  confidence_rationale: detailed explanation of why confidence is high/medium/low\n"
        "  risk_modifiers: array of features or caveats that increase/decrease risk\n"
        "  knowledge_highlights: array of short bullets extracted from curated knowledge hits\n"
        "  literature_highlights: array of short bullets extracted from literature hits\n"
        "  suggested_label: 'Toxic' or 'Non-toxic'\n"
        "  confidence: float 0.0-1.0\n"
        "Make the narrative explicit about evidence strength, disagreements, and whether Firestore or fallback sources were used.\n"
        f"Context JSON: {json.dumps(payload, ensure_ascii=True)}"
    )
