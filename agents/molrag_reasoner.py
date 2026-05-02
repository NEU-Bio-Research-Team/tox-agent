from __future__ import annotations

import json
import os
import re as _re
from typing import Any, Dict, List, Optional, Tuple

try:
    from google import genai
except Exception:
    genai = None

_MOLRAG_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "evidence_overview": {"type": "string"},
        "longform_summary": {"type": "string"},
        "mechanism_chain": {"type": "array", "items": {"type": "string"}},
        "key_substructures": {"type": "array", "items": {"type": "string"}},
        "confidence_rationale": {"type": "string"},
        "analogy_reasoning": {"type": "string"},
        "risk_modifiers": {"type": "array", "items": {"type": "string"}},
        "knowledge_highlights": {"type": "array", "items": {"type": "string"}},
        "literature_highlights": {"type": "array", "items": {"type": "string"}},
        "suggested_label": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["evidence_overview", "longform_summary", "mechanism_chain", "suggested_label", "confidence"],
}

from .language import choose_text, normalize_language
from services.knowledge_retriever import retrieve_knowledge_context
from services.prompt_builder import build_molrag_prompt
from services.genai_runtime import (
    build_genai_client_candidates,
    call_with_retry,
    dedupe_strings,
    is_model_unavailable_error,
    is_resource_exhausted_error,
)

MOLRAG_MODEL = os.getenv("AGENT_MODEL_FAST", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
MOLRAG_FALLBACK_MODEL = os.getenv("AGENT_MODEL_PRO", "gemini-2.5-pro")


def _safe_json_parse(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown code fences and repairing truncation."""
    text = text.strip()
    if text.startswith("```"):
        text = _re.sub(r"^```(?:json)?\n?", "", text).rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Attempt to repair truncated JSON: close any open string then close open objects/arrays
    repaired = text
    # Count open braces/brackets to figure out depth
    depth = 0
    in_string = False
    escape_next = False
    for ch in repaired:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == '{' or ch == '[':
                depth += 1
            elif ch == '}' or ch == ']':
                depth -= 1
    # If we're mid-string, close it first (truncated string value)
    if in_string:
        repaired += '"'
        depth_adjustment = depth
        # after closing the string we may be in the middle of a value; add null close
        repaired += ' }'
        depth_adjustment -= 1
        repaired += '}' * max(0, depth_adjustment - 1)
    else:
        repaired += '}' * max(0, depth)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    # Last resort: extract only complete top-level key-value pairs using regex
    result: dict = {}
    for m in _re.finditer(r'"(\w+)"\s*:\s*("(?:[^"\\]|\\.)*"|\[[^\]]*\]|[\d.]+|true|false|null)', text):
        key, val_str = m.group(1), m.group(2)
        try:
            result[key] = json.loads(val_str)
        except Exception:
            pass
    if result:
        return result
    raise ValueError(f"Could not parse LLM JSON response (length={len(text)})")


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

    # If analog coverage is weak but curated mechanism evidence is rich,
    # avoid over-penalizing known liabilities purely due to DB sparsity.
    if top_similarity < 0.2:
        mechanism_text = " ".join(
            [
                str(hit.get("name") or "").lower()
                + " "
                + str(hit.get("summary") or "").lower()
                for hit in knowledge_hits[:6]
            ]
        )
        has_known_liability = any(
            token in mechanism_text
            for token in ["herg", "qt", "torsades", "reactive metabolite", "mitochond", "dili"]
        )
        if has_known_liability:
            base += 0.08

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


def _build_key_substructures(
    knowledge_hits: List[Dict[str, Any]],
    retrieved_examples: List[Dict[str, Any]],
    input_smiles: str,
) -> List[str]:
    motifs: List[str] = []

    for hit in knowledge_hits:
        if not hit.get("smarts_hit"):
            continue
        name = str(hit.get("name") or "").strip()
        if name and name not in motifs:
            motifs.append(name)

    if not motifs:
        for hit in knowledge_hits[:3]:
            name = str(hit.get("name") or "").strip()
            if name and name not in motifs:
                motifs.append(name)

    if not motifs:
        for item in retrieved_examples[:2]:
            note = str(item.get("notes") or item.get("name") or "").strip()
            if note and note not in motifs:
                motifs.append(note[:80])

    smiles_lower = str(input_smiles or "").lower()
    # Lightweight heuristic for frequent aromatic oxygen scaffold mentions in antiarrhythmics.
    if any(token in smiles_lower for token in ["o1", "oc", "c1oc"]):
        benzofuran_label = "Benzofuran-like aromatic oxygen scaffold"
        if benzofuran_label not in motifs:
            motifs.append(benzofuran_label)

    return motifs[:5]


def _build_knowledge_highlights(knowledge_hits: List[Dict[str, Any]]) -> List[str]:
    highlights: List[str] = []
    for hit in knowledge_hits[:4]:
        name = str(hit.get("name") or "").strip() or "Unnamed mechanism"
        summary = str(hit.get("summary") or "").strip()
        risk = str(hit.get("risk_level") or "").strip().upper()
        prefix = f"{name}"
        if risk:
            prefix = f"{prefix} [{risk}]"
        text = f"{prefix}: {summary[:160]}" if summary else prefix
        highlights.append(text)
    return highlights


def _build_literature_highlights(literature_hits: List[Dict[str, Any]]) -> List[str]:
    highlights: List[str] = []
    for hit in literature_hits[:4]:
        title = str(hit.get("title") or "").strip() or "Untitled paper"
        excerpt = str(hit.get("excerpt") or "").strip()
        year = str(hit.get("year") or "").strip()
        pmid = str(hit.get("pmid") or "").strip()
        ref_parts = [part for part in [year, f"PMID:{pmid}" if pmid else ""] if part]
        ref = f" ({', '.join(ref_parts)})" if ref_parts else ""
        highlights.append(f"{title}{ref}: {excerpt[:160]}" if excerpt else f"{title}{ref}")
    return highlights


def _build_analogy_reasoning(
    *,
    retrieved_examples: List[Dict[str, Any]],
    suggested_label: str,
    baseline_label: str,
    contrastive_pair: Optional[Dict[str, Any]],
    language: str,
) -> str:
    if not retrieved_examples:
        return choose_text(
            language,
            "Khong co analog dat nguong similarity, nen MolRAG khong co du bang chung so sanh truc tiep va phai dua nhieu hon vao knowledge/literature bo tro.",
            "No analog passed the similarity threshold, so MolRAG lacks direct structural comparators and must rely more heavily on supporting knowledge and literature.",
        )

    top_analog = max(retrieved_examples, key=lambda item: float(item.get("similarity", 0.0) or 0.0))
    top_name = str(top_analog.get("name") or top_analog.get("smiles") or "top analog").strip()
    top_similarity = float(top_analog.get("similarity", 0.0) or 0.0)
    top_label = str(top_analog.get("label") or "Unknown").strip()

    base_text = choose_text(
        language,
        f"Analog gan nhat la {top_name} voi similarity={top_similarity:.2f} va nhan {top_label}; xu huong nay dan MolRAG nghieng ve {suggested_label} trong khi baseline hien tai la {baseline_label}.",
        f"The closest analog is {top_name} with similarity={top_similarity:.2f} and label {top_label}; this pushes MolRAG toward {suggested_label} while the current baseline remains {baseline_label}.",
    )

    if contrastive_pair and contrastive_pair.get("note"):
        return f"{base_text} {contrastive_pair['note']}"
    return base_text


def _build_confidence_rationale(
    *,
    confidence_zone: str,
    top_similarity: float,
    has_smarts_hit: bool,
    knowledge_hits: List[Dict[str, Any]],
    literature_hits: List[Dict[str, Any]],
    firestore_state: Dict[str, Any],
    language: str,
) -> str:
    firestore_ready = bool(firestore_state.get("ready", False))
    firestore_note = choose_text(
        language,
        "Firestore san sang" if firestore_ready else "Firestore khong san sang, mot phan bang chung co the dang dung fallback",
        "Firestore is ready" if firestore_ready else "Firestore is not ready, so some evidence may be coming from fallback sources",
    )
    smarts_note = choose_text(
        language,
        "co SMARTS hit" if has_smarts_hit else "khong co SMARTS hit truc tiep",
        "a direct SMARTS hit is present" if has_smarts_hit else "no direct SMARTS hit is present",
    )

    return choose_text(
        language,
        (
            f"Confidence zone={confidence_zone}, similarity cao nhat={top_similarity:.2f}, {smarts_note}, "
            f"knowledge_hits={len(knowledge_hits)}, literature_hits={len(literature_hits)}; {firestore_note}."
        ),
        (
            f"Confidence zone={confidence_zone}, top similarity={top_similarity:.2f}, {smarts_note}, "
            f"knowledge_hits={len(knowledge_hits)}, literature_hits={len(literature_hits)}; {firestore_note}."
        ),
    )


def _build_risk_modifiers(
    *,
    knowledge_hits: List[Dict[str, Any]],
    literature_hits: List[Dict[str, Any]],
    top_similarity: float,
    has_smarts_hit: bool,
    firestore_state: Dict[str, Any],
    language: str,
) -> List[str]:
    modifiers: List[str] = []
    if top_similarity < 0.5:
        modifiers.append(
            choose_text(
                language,
                "Similarity thap lam giam do tin cay khi suy dien tu analog.",
                "Low similarity weakens confidence in analog-based extrapolation.",
            )
        )
    if has_smarts_hit:
        modifiers.append(
            choose_text(
                language,
                "Co SMARTS hit truc tiep lam tang suc nang co che.",
                "A direct SMARTS hit strengthens mechanistic plausibility.",
            )
        )
    if any(str(hit.get("risk_level") or "").strip().lower() in {"high", "severe"} for hit in knowledge_hits):
        modifiers.append(
            choose_text(
                language,
                "Knowledge base chua cac motif nguy co cao, day la tin hieu tang risk.",
                "Curated knowledge contains high-risk motifs, which increases concern.",
            )
        )
    if not literature_hits:
        modifiers.append(
            choose_text(
                language,
                "Khong co literature hit manh nen lop bang chung van con mong.",
                "No strong literature hits were retrieved, so the evidence layer remains thin.",
            )
        )
    if not bool(firestore_state.get("ready", False)):
        modifiers.append(
            choose_text(
                language,
                "Firestore chua san sang, co the dang dung csv/doc fallback thay vi kho tri thuc day du.",
                "Firestore is not ready, so fallback data may be used instead of the full curated store.",
            )
        )
    return modifiers[:5]


def _compose_longform_summary(
    *,
    baseline_label: str,
    suggested_label: str,
    top_similarity: float,
    toxic_score: float,
    non_toxic_score: float,
    knowledge_hits: List[Dict[str, Any]],
    literature_hits: List[Dict[str, Any]],
    contrastive_pair: Optional[Dict[str, Any]],
    confidence_rationale: str,
    firestore_state: Dict[str, Any],
    language: str,
) -> str:
    top_knowledge = ", ".join(str(hit.get("name") or "").strip() for hit in knowledge_hits[:2] if str(hit.get("name") or "").strip()) or choose_text(language, "khong co", "none")
    top_literature = ", ".join(str(hit.get("title") or "").strip() for hit in literature_hits[:2] if str(hit.get("title") or "").strip()) or choose_text(language, "khong co", "none")
    contrast_note = str(contrastive_pair.get("note") or "").strip() if contrastive_pair else ""
    firestore_source = choose_text(
        language,
        "Firestore san sang" if firestore_state.get("ready") else "Firestore khong san sang",
        "Firestore is ready" if firestore_state.get("ready") else "Firestore is not ready",
    )

    return choose_text(
        language,
        (
            f"MolRAG tim thay bang chung tu analog voi top similarity={top_similarity:.2f}, toxic_score={toxic_score:.2f} va non_toxic_score={non_toxic_score:.2f}. "
            f"Lop knowledge/literature hien tai nhan manh {top_knowledge}; cac paper noi bat gom {top_literature}. "
            f"Do do, MolRAG tam nghieng ve nhan {suggested_label}, nhung baseline van giu vai tro quyet dinh cuoi la {baseline_label}. "
            f"{contrast_note} {confidence_rationale} {firestore_source}."
        ),
        (
            f"MolRAG retrieved analog evidence with top similarity={top_similarity:.2f}, toxic_score={toxic_score:.2f}, and non_toxic_score={non_toxic_score:.2f}. "
            f"The current knowledge/literature layer emphasizes {top_knowledge}; top papers include {top_literature}. "
            f"As a result, MolRAG currently leans toward {suggested_label}, while the baseline remains the final decision source as {baseline_label}. "
            f"{contrast_note} {confidence_rationale} {firestore_source}."
        ),
    )


def _confidence_label(confidence: float, language: str) -> str:
    if confidence >= 0.75:
        return choose_text(language, "Tin cậy cao", "High confidence")
    if confidence >= 0.45:
        return choose_text(language, "Tin cậy trung bình", "Moderate confidence")
    return choose_text(language, "Tin cậy thấp", "Low confidence")


def _top_analog(retrieved_examples: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not retrieved_examples:
        return None
    return max(
        retrieved_examples,
        key=lambda item: float(item.get("similarity", 0.0) or 0.0),
    )


def _append_if_present(items: List[str], value: str) -> None:
    cleaned = str(value or "").strip()
    if cleaned and cleaned not in items:
        items.append(cleaned)


def _build_presentation(
    *,
    baseline_label: str,
    suggested_label: str,
    confidence: float,
    top_similarity: float,
    retrieved_examples: List[Dict[str, Any]],
    knowledge_hits: List[Dict[str, Any]],
    literature_hits: List[Dict[str, Any]],
    confidence_rationale: str,
    risk_modifiers: List[str],
    language: str,
) -> Dict[str, Any]:
    top_analog = _top_analog(retrieved_examples)
    top_knowledge = knowledge_hits[0] if knowledge_hits else None
    top_literature = literature_hits[0] if literature_hits else None
    disagrees = suggested_label.strip().lower() != baseline_label.strip().lower()
    confidence_label = _confidence_label(confidence, language)

    if disagrees:
        headline = choose_text(
            language,
            f"MolRAG nghiêng về {suggested_label}, nhưng baseline vẫn chốt {baseline_label}.",
            f"MolRAG leans {suggested_label}, while the baseline still decides {baseline_label}.",
        )
    else:
        headline = choose_text(
            language,
            f"MolRAG và baseline đang đồng thuận ở nhãn {baseline_label}.",
            f"MolRAG and the baseline are aligned on the {baseline_label} label.",
        )

    subheadline = choose_text(
        language,
        f"Top analog similarity {top_similarity:.2f}; {confidence_label.lower()}; phần kết luận cuối hiện vẫn ưu tiên baseline trong chế độ MVP.",
        f"Top analog similarity is {top_similarity:.2f}; {confidence_label.lower()}; the final call still prioritizes the baseline in MVP mode.",
    )

    takeaways: List[str] = []
    if top_analog is not None:
        top_name = str(top_analog.get("name") or top_analog.get("smiles") or "top analog").strip()
        top_label = str(top_analog.get("label") or "Unknown").strip()
        _append_if_present(
            takeaways,
            choose_text(
                language,
                f"Analog gần nhất là {top_name} (similarity {float(top_analog.get('similarity', 0.0) or 0.0):.2f}) với nhãn {top_label}.",
                f"The closest analog is {top_name} (similarity {float(top_analog.get('similarity', 0.0) or 0.0):.2f}) with label {top_label}.",
            ),
        )
    else:
        _append_if_present(
            takeaways,
            choose_text(
                language,
                "Không có analog vượt ngưỡng similarity, nên MolRAG phải dựa nhiều hơn vào lớp knowledge/literature.",
                "No analog cleared the similarity threshold, so MolRAG leans more heavily on knowledge and literature.",
            ),
        )

    if top_knowledge is not None:
        _append_if_present(
            takeaways,
            choose_text(
                language,
                f"Tín hiệu cơ chế nổi bật nhất là {str(top_knowledge.get('name') or 'N/A').strip()}.",
                f"The strongest mechanistic signal is {str(top_knowledge.get('name') or 'N/A').strip()}.",
            ),
        )

    if top_literature is not None:
        _append_if_present(
            takeaways,
            choose_text(
                language,
                f"Literature hỗ trợ hiện thiên về bài: {str(top_literature.get('title') or 'N/A').strip()}.",
                f"Current literature support is led by: {str(top_literature.get('title') or 'N/A').strip()}.",
            ),
        )

    _append_if_present(takeaways, confidence_rationale)

    evidence_cards: List[Dict[str, Any]] = []
    if top_analog is not None:
        evidence_cards.append(
            {
                "eyebrow": choose_text(language, "Analog chính", "Top analog"),
                "title": str(top_analog.get("name") or top_analog.get("smiles") or "Top analog").strip(),
                "body": choose_text(
                    language,
                    f"Similarity {float(top_analog.get('similarity', 0.0) or 0.0):.2f} · nhãn {str(top_analog.get('label') or 'Unknown').strip()}.",
                    f"Similarity {float(top_analog.get('similarity', 0.0) or 0.0):.2f} · label {str(top_analog.get('label') or 'Unknown').strip()}.",
                ),
                "tone": "conflict" if disagrees else "support",
            }
        )

    if top_knowledge is not None:
        evidence_cards.append(
            {
                "eyebrow": choose_text(language, "Cơ chế", "Mechanism"),
                "title": str(top_knowledge.get("name") or "Mechanistic signal").strip(),
                "body": str(top_knowledge.get("summary") or "").strip()[:180],
                "tone": "warning" if str(top_knowledge.get("risk_level") or "").strip().lower() in {"high", "severe"} else "neutral",
            }
        )

    if top_literature is not None:
        evidence_cards.append(
            {
                "eyebrow": choose_text(language, "Literature", "Literature"),
                "title": str(top_literature.get("title") or "Literature support").strip(),
                "body": str(top_literature.get("excerpt") or "").strip()[:180],
                "tone": "neutral",
            }
        )

    if disagrees or risk_modifiers:
        evidence_cards.append(
            {
                "eyebrow": choose_text(language, "Cảnh báo", "Caveat"),
                "title": choose_text(language, "Điểm cần thận trọng", "What to watch"),
                "body": risk_modifiers[0] if risk_modifiers else choose_text(
                    language,
                    "MolRAG và baseline chưa đồng thuận hoàn toàn, nên cần đọc kết quả như lớp bằng chứng bổ sung.",
                    "MolRAG and the baseline are not fully aligned, so treat this as supporting evidence rather than the final decision.",
                ),
                "tone": "warning",
            }
        )

    caveats = risk_modifiers[:3]
    if disagrees:
        caveats.insert(
            0,
            choose_text(
                language,
                f"MolRAG ({suggested_label}) đang khác baseline ({baseline_label}).",
                f"MolRAG ({suggested_label}) is currently different from the baseline ({baseline_label}).",
            ),
        )

    return {
        "headline": headline,
        "subheadline": subheadline,
        "takeaways": takeaways[:4],
        "evidence_cards": evidence_cards[:4],
        "caveats": caveats[:3],
        "confidence_banner": {
            "label": confidence_label,
            "detail": confidence_rationale,
        },
    }


def _deterministic_reasoning(
    *,
    input_smiles: str,
    retrieved_examples: List[Dict[str, Any]],
    baseline_prediction: Dict[str, Any],
    knowledge_context: Dict[str, Any],
    language: str,
    strategy: str,
    retrieval_context: Optional[Dict[str, Any]] = None,
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
    firestore_state = knowledge_context.get("firestore") if isinstance(knowledge_context.get("firestore"), dict) else {}
    retrieval_context = retrieval_context if isinstance(retrieval_context, dict) else {}
    key_substructures = _build_key_substructures(knowledge_hits, retrieved_examples, input_smiles)
    knowledge_highlights = _build_knowledge_highlights(knowledge_hits)
    literature_highlights = _build_literature_highlights(literature_hits)
    analogy_reasoning = _build_analogy_reasoning(
        retrieved_examples=retrieved_examples,
        suggested_label=suggested_label,
        baseline_label=baseline_label,
        contrastive_pair=contrastive_pair,
        language=language,
    )
    confidence_rationale = _build_confidence_rationale(
        confidence_zone=confidence_zone,
        top_similarity=top_similarity,
        has_smarts_hit=has_smarts_hit,
        knowledge_hits=knowledge_hits,
        literature_hits=literature_hits,
        firestore_state=firestore_state,
        language=language,
    )
    risk_modifiers = _build_risk_modifiers(
        knowledge_hits=knowledge_hits,
        literature_hits=literature_hits,
        top_similarity=top_similarity,
        has_smarts_hit=has_smarts_hit,
        firestore_state=firestore_state,
        language=language,
    )

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
    evidence_overview = choose_text(
        language,
        (
            f"Nguon retrieval={retrieval_context.get('db_source') or 'unknown'}, db_size={retrieval_context.get('db_size') or 0}, "
            f"firestore_ready={bool(firestore_state.get('ready', False))}. {evidence_summary}"
        ),
        (
            f"Retrieval source={retrieval_context.get('db_source') or 'unknown'}, db_size={retrieval_context.get('db_size') or 0}, "
            f"firestore_ready={bool(firestore_state.get('ready', False))}. {evidence_summary}"
        ),
    )

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

    longform_summary = _compose_longform_summary(
        baseline_label=baseline_label,
        suggested_label=suggested_label,
        top_similarity=top_similarity,
        toxic_score=toxic_score,
        non_toxic_score=non_toxic_score,
        knowledge_hits=knowledge_hits,
        literature_hits=literature_hits,
        contrastive_pair=contrastive_pair,
        confidence_rationale=confidence_rationale,
        firestore_state=firestore_state,
        language=language,
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
    presentation = _build_presentation(
        baseline_label=baseline_label,
        suggested_label=suggested_label,
        confidence=confidence,
        top_similarity=top_similarity,
        retrieved_examples=retrieved_examples,
        knowledge_hits=knowledge_hits,
        literature_hits=literature_hits,
        confidence_rationale=confidence_rationale,
        risk_modifiers=risk_modifiers,
        language=language,
    )

    return {
        "enabled": True,
        "strategy": strategy,
        "input_smiles": input_smiles,
        "reasoning_mode": "deterministic",
        "evidence_overview": evidence_overview,
        "evidence_summary": evidence_summary,
        "reasoning_summary": reasoning_summary,
        "longform_summary": longform_summary,
        "mechanism_chain": mechanism_chain,
        "key_substructures": key_substructures,
        "analogy_reasoning": analogy_reasoning,
        "confidence_rationale": confidence_rationale,
        "risk_modifiers": risk_modifiers,
        "knowledge_highlights": knowledge_highlights,
        "literature_highlights": literature_highlights,
        "presentation": presentation,
        "suggested_label": suggested_label,
        "confidence": confidence,
        "confidence_zone": confidence_zone,
        "tox_classes": tox_classes,
        "knowledge_hits": knowledge_hits,
        "literature_hits": literature_hits,
        "molrag_evidence": molrag_evidence,
        "retrieval_overview": {
            "db_source": retrieval_context.get("db_source"),
            "db_size": retrieval_context.get("db_size"),
            "match_count": len(retrieved_examples),
        },
        "knowledge_error": knowledge_context.get("error"),
        "firestore": firestore_state,
    }


def run_molrag_reasoning(
    *,
    input_smiles: str,
    retrieved_examples: List[Dict[str, Any]],
    baseline_prediction: Dict[str, Any],
    language: str = "vi",
    strategy: str = "sim_cot",
    retrieval_context: Optional[Dict[str, Any]] = None,
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
        retrieval_context=retrieval_context,
        firestore_state=knowledge_context.get("firestore") if isinstance(knowledge_context.get("firestore"), dict) else {},
        strategy=strategy,
    )

    result = _deterministic_reasoning(
        input_smiles=input_smiles,
        retrieved_examples=retrieved_examples,
        baseline_prediction=baseline_prediction,
        knowledge_context=knowledge_context,
        language=normalized_language,
        strategy=strategy,
        retrieval_context=retrieval_context,
    )
    result["prompt_preview"] = prompt[:1800]

    if genai is None or not MOLRAG_MODEL:
        result["llm_status"] = "llm_unavailable"
        return result

    client_candidates = build_genai_client_candidates()
    if not client_candidates:
        result["llm_status"] = "llm_client_unavailable"
        return result

    model_candidates = dedupe_strings([MOLRAG_MODEL, MOLRAG_FALLBACK_MODEL])
    config = genai.types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.3,
        max_output_tokens=8192,
        response_schema=_MOLRAG_RESPONSE_SCHEMA,
    )

    errors: List[str] = []
    for client, auth_mode in client_candidates:
        for model_name in model_candidates:
            try:
                response = call_with_retry(
                    lambda client=client, model_name=model_name: client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )
                )
                llm_out = _safe_json_parse(response.text)
                result.update(llm_out)
                result["reasoning_mode"] = "llm"
                result["llm_status"] = f"llm_ok:{auth_mode}:{model_name}"
                return result
            except Exception as exc:
                errors.append(f"llm_error:{type(exc).__name__}:{auth_mode}:{model_name}:{str(exc)[:180]}")
                if is_resource_exhausted_error(exc) or is_model_unavailable_error(exc):
                    continue
                result["llm_status"] = errors[0]
                return result

    result["llm_status"] = errors[0] if errors else "llm_error:unknown"

    return result
