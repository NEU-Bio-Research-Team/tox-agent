from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence, Set

from .firestore_client import fetch_collection_documents, get_firestore_availability


_TEXT_HINT_KEYWORDS: Dict[str, Sequence[str]] = {
    "herg_inhibitor": ("herg", "qt", "arrhythmia", "torsades"),
    "hepatotoxic": ("liver", "hepat", "dili", "alt", "ast"),
    "reactive_metabolite": ("reactive", "metabolite", "bioactivation", "covalent"),
    "oxidative_stress": ("oxidative", "mitochondria"),
    "genotoxic": ("genotoxic", "mutagenic", "ames"),
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "null", "nan"} else text


def _to_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    return []


def _extract_tox_classes(retrieved_examples: Sequence[Dict[str, Any]]) -> List[str]:
    classes: Set[str] = set()

    for item in retrieved_examples:
        for value in _to_list(item.get("tox_class")):
            classes.add(value.lower())

        label = _clean_text(item.get("label")).lower()
        if label.startswith("toxic"):
            classes.add("general_toxicity")

    return sorted(classes)


def _derive_keyword_hints(tox_classes: Sequence[str]) -> List[str]:
    hints: Set[str] = set()
    for tox_class in tox_classes:
        hints.add(tox_class)
        for keyword in _TEXT_HINT_KEYWORDS.get(tox_class, ()):  # keep hints broad
            hints.add(keyword)
    return sorted(hints)


@lru_cache(maxsize=1)
def _load_knowledge_docs() -> List[Dict[str, Any]]:
    return fetch_collection_documents("molrag_knowledge")


@lru_cache(maxsize=1)
def _load_literature_docs() -> List[Dict[str, Any]]:
    return fetch_collection_documents("molrag_literature")


def _score_knowledge_doc(doc: Dict[str, Any], tox_classes: Sequence[str], keyword_hints: Sequence[str]) -> float:
    score = 0.0

    doc_tox = {item.lower() for item in _to_list(doc.get("tox_class"))}
    overlap = doc_tox.intersection({value.lower() for value in tox_classes})
    score += float(len(overlap)) * 2.0

    haystack = " ".join(
        [
            _clean_text(doc.get("name")),
            _clean_text(doc.get("title")),
            _clean_text(doc.get("summary")),
            _clean_text(doc.get("clinical_manifestation")),
            _clean_text(doc.get("smarts")),
        ]
    ).lower()

    for hint in keyword_hints:
        if hint and hint.lower() in haystack:
            score += 1.0

    if _clean_text(doc.get("type")).lower() == "mechanism":
        score += 0.5

    return score


def _score_literature_doc(doc: Dict[str, Any], keyword_hints: Sequence[str]) -> float:
    score = 0.0
    haystack = " ".join(
        [
            _clean_text(doc.get("title")),
            _clean_text(doc.get("abstract_chunk")),
            _clean_text(doc.get("source_query")),
            " ".join(_to_list(doc.get("relevant_targets"))),
            " ".join(_to_list(doc.get("compound_mentions"))),
        ]
    ).lower()

    for hint in keyword_hints:
        if hint and hint.lower() in haystack:
            score += 1.0

    try:
        score += min(1.5, max(0.0, (float(doc.get("year", 0)) - 2010.0) / 20.0))
    except Exception:
        pass

    return score


def retrieve_knowledge_context(
    *,
    input_smiles: str,
    retrieved_examples: Sequence[Dict[str, Any]],
    top_k_knowledge: int = 4,
    top_k_literature: int = 4,
) -> Dict[str, Any]:
    del input_smiles  # reserved for future embedding/rule-based enrichments

    firestore_state = get_firestore_availability()
    knowledge_docs = _load_knowledge_docs()
    literature_docs = _load_literature_docs()
    tox_classes = _extract_tox_classes(retrieved_examples)
    keyword_hints = _derive_keyword_hints(tox_classes)

    if not knowledge_docs and not literature_docs:
        return {
            "tox_classes": tox_classes,
            "keyword_hints": keyword_hints,
            "knowledge_hits": [],
            "literature_hits": [],
            "error": "knowledge_store_empty",
            "firestore": firestore_state,
        }

    knowledge_scored: List[Dict[str, Any]] = []
    for doc in knowledge_docs:
        score = _score_knowledge_doc(doc, tox_classes, keyword_hints)
        if score <= 0:
            continue

        knowledge_scored.append(
            {
                "doc_id": _clean_text(doc.get("doc_id")),
                "type": _clean_text(doc.get("type")),
                "name": _clean_text(doc.get("name") or doc.get("title")),
                "summary": _clean_text(doc.get("summary")),
                "tox_class": _to_list(doc.get("tox_class")),
                "risk_level": _clean_text(doc.get("risk_level") or doc.get("severity")),
                "score": round(score, 3),
            }
        )

    knowledge_scored.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    literature_scored: List[Dict[str, Any]] = []
    for doc in literature_docs:
        score = _score_literature_doc(doc, keyword_hints)
        if score <= 0:
            continue

        literature_scored.append(
            {
                "doc_id": _clean_text(doc.get("doc_id")),
                "title": _clean_text(doc.get("title")),
                "year": doc.get("year"),
                "source_query": _clean_text(doc.get("source_query")),
                "relevant_targets": _to_list(doc.get("relevant_targets")),
                "compound_mentions": _to_list(doc.get("compound_mentions"))[:5],
                "excerpt": _clean_text(doc.get("abstract_chunk"))[:280],
                "score": round(score, 3),
            }
        )

    literature_scored.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            int(item.get("year", 0) or 0),
        ),
        reverse=True,
    )

    return {
        "tox_classes": tox_classes,
        "keyword_hints": keyword_hints,
        "knowledge_hits": knowledge_scored[: max(int(top_k_knowledge), 0)],
        "literature_hits": literature_scored[: max(int(top_k_literature), 0)],
        "error": None,
        "firestore": firestore_state,
    }
