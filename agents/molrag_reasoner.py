from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

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


def _weighted_vote(retrieved_examples: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Similarity-weighted vote: returns (toxic_score, non_toxic_score)."""
    toxic_score = 0.0
    non_toxic_score = 0.0
    for item in retrieved_examples:
        label = str(item.get("label") or "").strip().lower()
        weight = float(item.get("similarity") or 0.5)
        if "non" in label or "safe" in label:
            non_toxic_score += weight
        elif "toxic" in label or label == "1":
            toxic_score += weight
    return toxic_score, non_toxic_score


def _calibrate_confidence(
    top_similarity: float,
    toxic_score: float,
    non_toxic_score: float,
    has_smarts_hit: bool,
    knowledge_hits: List[Dict[str, Any]],
    literature_hits: List[Dict[str, Any]],
) -> Tuple[float, str]:
    """Calibrate confidence based on chemical space distance and evidence quality."""
    vote_total = toxic_score + non_toxic_score
    vote_margin = abs(toxic_score - non_toxic_score) / max(vote_total, 0.01)

    if top_similarity >= 0.9:
        base = 0.80 + vote_margin * 0.15
        zone = "high_confidence"
    elif top_similarity >= 0.7:
        base = 0.55 + vote_margin * 0.20
        zone = "medium_confidence"
    elif top_similarity >= 0.5:
        base = 0.35 + vote_margin * 0.15
        zone = "low_confidence"
    else:
        base = 0.20 + vote_margin * 0.10
        zone = "extrapolation_zone"

    if has_smarts_hit:
        base += 0.05
    base += min(len(knowledge_hits), 4) * 0.01
    base += min(len(literature_hits), 4) * 0.005

    return round(min(0.95, base), 3), zone


def _build_mechanism_chain(
    knowledge_hits: List[Dict[str, Any]],
    literature_hits: List[Dict[str, Any]],
    toxic_score: float,
    non_toxic_score: float,
    suggested_label: str,
    top_similarity: float,
) -> List[str]:
    """Build a chain-of-thought reasoning list from available evidence."""
    chain: List[str] = []

    # SMARTS hits first — most mechanistically specific
    for h in [h for h in knowledge_hits if h.get("smarts_hit")][:2]:
        chain.append(f"SMARTS match: {h['name']} — {h.get('summary', '')[:120]}")

    # Non-SMARTS mechanism knowledge hits
    for h in [h for h in knowledge_hits if not h.get("smarts_hit")][:2]:
        risk = h.get("risk_level", "")
        risk_note = f" [risk: {risk}]" if risk else ""
        chain.append(f"Mechanism: {h['name']}{risk_note} — {h.get('summary', '')[:100]}")

    # Weighted analog vote summary
    vote_total = toxic_score + non_toxic_score
    if vote_total > 0:
        chain.append(
            f"Analog vote: {toxic_score:.2f} toxic-weighted / {non_toxic_score:.2f} non-toxic-weighted "
            f"→ leans {suggested_label} (top similarity={top_similarity:.2f})"
        )

    # Literature reference
    for lit in literature_hits[:1]:
        year = lit.get("year", "")
        pmid = lit.get("pmid", "")
        ref = f" [PMID:{pmid}]" if pmid else (f" ({year})" if year else "")
        chain.append(f"Literature: {lit.get('title', '')[:120]}{ref}")

    if not chain:
        chain.append("No direct mechanism evidence found; baseline verdict retained.")

    return chain


def _find_contrastive_pair(
    retrieved_examples: List[Dict[str, Any]],
    suggested_label: str,
) -> Optional[Dict[str, Any]]:
    """Find the closest analog with the OPPOSITE label for contrastive reasoning."""
    target_norm = _normalize_label(suggested_label)
    opposite_norm = "NON_TOXIC" if target_norm == "TOXIC" else "TOXIC"

    best_opposite = max(
        (item for item in retrieved_examples if _normalize_label(item.get("label", "")) == opposite_norm),
        key=lambda x: float(x.get("similarity", 0.0) or 0.0),
        default=None,
    )
    best_same = max(
        (item for item in retrieved_examples if _normalize_label(item.get("label", "")) == target_norm),
        key=lambda x: float(x.get("similarity", 0.0) or 0.0),
        default=None,
    )

    if best_opposite is None or best_same is None:
        return None

    opposite_sim = float(best_opposite.get("similarity", 0.0) or 0.0)
    if opposite_sim < 0.6:
        return None

    same_sim = float(best_same.get("similarity", 0.0) or 0.0)
    return {
        "same_label_analog": best_same.get("smiles", ""),
        "same_label_name": best_same.get("name", ""),
        "same_label": best_same.get("label", ""),
        "same_sim": round(same_sim, 3),
        "opposite_label_analog": best_opposite.get("smiles", ""),
        "opposite_label_name": best_opposite.get("name", ""),
        "opposite_label": best_opposite.get("label", ""),
        "opposite_sim": round(opposite_sim, 3),
        "note": (
            f"'{best_same.get('name', best_same.get('smiles', ''))[:60]}' "
            f"(sim={same_sim:.2f}) → {best_same.get('label', 'unknown')}, "
            f"vs '{best_opposite.get('name', best_opposite.get('smiles', ''))[:60]}' "
            f"(sim={opposite_sim:.2f}) → {best_opposite.get('label', 'unknown')}. "
            "Structural delta may explain the label difference."
        ),
    }


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
    has_smarts_hit = bool(knowledge_context.get("has_smarts_hit", False))

    toxic_score, non_toxic_score = _weighted_vote(retrieved_examples)
    top_similarity = max((float(item.get("similarity", 0.0) or 0.0) for item in retrieved_examples), default=0.0)
    high_risk_mechanisms = sum(
        1
        for item in knowledge_hits
        if str(item.get("risk_level") or "").strip().lower() in {"high", "severe"}
    )

    if toxic_score > non_toxic_score:
        suggested_label = "Toxic"
    elif non_toxic_score > toxic_score:
        suggested_label = "Non-toxic"
    else:
        suggested_label = baseline_label

    confidence, confidence_zone = _calibrate_confidence(
        top_similarity=top_similarity,
        toxic_score=toxic_score,
        non_toxic_score=non_toxic_score,
        has_smarts_hit=has_smarts_hit,
        knowledge_hits=knowledge_hits,
        literature_hits=literature_hits,
    )

    mechanism_chain = _build_mechanism_chain(
        knowledge_hits=knowledge_hits,
        literature_hits=literature_hits,
        toxic_score=toxic_score,
        non_toxic_score=non_toxic_score,
        suggested_label=suggested_label,
        top_similarity=top_similarity,
    )

    contrastive_pair = _find_contrastive_pair(retrieved_examples, suggested_label)

    analog_evidence = choose_text(
        language,
        (
            f"Tim thay {len(retrieved_examples)} analog, top similarity={top_similarity:.2f}, "
            f"toxic_score={toxic_score:.2f}, non_toxic_score={non_toxic_score:.2f}."
        ),
        (
            f"Retrieved {len(retrieved_examples)} analogs with top similarity={top_similarity:.2f}, "
            f"toxic_score={toxic_score:.2f}, non_toxic_score={non_toxic_score:.2f}."
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

    # Structured evidence block for writer_agent rendering
    molrag_evidence = {
        "analog_support": {
            "count": len(retrieved_examples),
            "top_sim": round(top_similarity, 3),
            "top_name": next(
                (item.get("name", item.get("smiles", "")) for item in sorted(
                    retrieved_examples, key=lambda x: float(x.get("similarity", 0.0) or 0.0), reverse=True
                )),
                "",
            ),
            "vote": f"{toxic_score:.2f} toxic / {non_toxic_score:.2f} non-toxic",
        },
        "mechanism_matches": [
            {
                "name": h.get("name", ""),
                "smarts_hit": bool(h.get("smarts_hit")),
                "risk": h.get("risk_level", ""),
                "summary": h.get("summary", "")[:120],
            }
            for h in knowledge_hits[:4]
        ],
        "literature_support": [
            {
                "title": lit.get("title", ""),
                "year": lit.get("year"),
                "pmid": lit.get("pmid", ""),
                "excerpt": lit.get("excerpt", "")[:200],
            }
            for lit in literature_hits[:3]
        ],
        "contrastive_pair": contrastive_pair,
        "confidence_zone": confidence_zone,
        "has_smarts_hit": has_smarts_hit,
    }

    return {
        "enabled": True,
        "strategy": strategy,
        "input_smiles": input_smiles,
        "reasoning_mode": "deterministic",
        "evidence_summary": evidence_summary,
        "reasoning_summary": reasoning_summary,
        "mechanism_chain": mechanism_chain,
        "suggested_label": suggested_label,
        "confidence": confidence,
        "confidence_zone": confidence_zone,
        "tox_classes": tox_classes,
        "knowledge_hits": knowledge_hits,
        "literature_hits": literature_hits,
        "molrag_evidence": molrag_evidence,
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

    if genai is None or not os.getenv("GEMINI_MODEL"):
        result["llm_status"] = "llm_unavailable"
        return result

    try:
        client = genai.Client()
        response = client.models.generate_content(
            model=MOLRAG_MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "mechanism_chain": {"type": "array", "items": {"type": "string"}},
                        "key_substructures": {"type": "array", "items": {"type": "string"}},
                        "confidence_rationale": {"type": "string"},
                        "analogy_reasoning": {"type": "string"},
                        "risk_modifiers": {"type": "array", "items": {"type": "string"}},
                        "suggested_label": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["mechanism_chain", "suggested_label", "confidence"],
                },
            ),
        )
        llm_out = json.loads(response.text)
        # Merge LLM output — LLM mechanism_chain overrides deterministic when available
        result.update(llm_out)
        result["reasoning_mode"] = "llm"
        result["llm_status"] = "llm_ok"
    except Exception as exc:
        result["llm_status"] = f"llm_error: {exc!s}"

    return result
