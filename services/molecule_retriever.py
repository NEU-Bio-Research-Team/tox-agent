from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from .firestore_client import fetch_collection_documents, get_firestore_availability
from .fingerprint_service import canonicalize_smiles, fingerprint_from_smiles, tanimoto_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RETRIEVAL_FILES = [
    PROJECT_ROOT / "test_data" / "reference_panel.csv",
    PROJECT_ROOT / "test_data" / "screening_library.csv",
    PROJECT_ROOT / "test_data" / "full_test_set.csv",
    PROJECT_ROOT / "test_data" / "toxic_compounds.csv",
    PROJECT_ROOT / "test_data" / "smiles_only.csv",
]


def _normalize_label(raw_label: Any) -> str:
    text = str(raw_label or "").strip().lower()
    if text in {"1", "1.0", "true", "toxic", "positive", "high"}:
        return "Toxic"
    if text in {"0", "0.0", "false", "safe", "non-toxic", "non_toxic", "negative", "low"}:
        return "Non-toxic"
    return str(raw_label or "Unknown").strip() or "Unknown"


def _build_entry(
    *,
    entry_id: str,
    name: str,
    smiles: str,
    label: Any,
    source_dataset: str,
    notes: str,
    tox_class: Any,
) -> Dict[str, Any] | None:
    validated = canonicalize_smiles(smiles)
    if not validated.get("valid"):
        return None

    canonical_smiles = str(validated["canonical_smiles"])
    fingerprint = fingerprint_from_smiles(canonical_smiles)
    if fingerprint is None:
        return None

    return {
        "entry_id": entry_id,
        "name": name or canonical_smiles,
        "smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "label": _normalize_label(label),
        "source_dataset": source_dataset,
        "notes": notes,
        "tox_class": tox_class if isinstance(tox_class, list) else [],
        "fingerprint": fingerprint,
    }


def _load_firestore_database() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    documents = fetch_collection_documents("molrag_compounds")
    if not documents:
        return rows

    for document in documents:
        smiles = str(document.get("smiles") or document.get("canonical_smiles") or "").strip()
        if not smiles:
            continue

        entry = _build_entry(
            entry_id=str(document.get("compound_id") or document.get("doc_id") or "").strip() or str(document.get("canonical_smiles") or smiles),
            name=str(document.get("common_name") or document.get("name") or document.get("iupac_name") or "").strip(),
            smiles=smiles,
            label=document.get("label"),
            source_dataset=str(document.get("source_dataset") or "molrag_compounds"),
            notes=str(document.get("notes") or "").strip(),
            tox_class=document.get("tox_class") or [],
        )
        if entry is not None:
            rows.append(entry)

    return rows


def _load_csv_database() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for csv_path in DEFAULT_RETRIEVAL_FILES:
        if not csv_path.exists():
            continue

        with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                smiles = str(row.get("smiles") or "").strip()
                if not smiles:
                    continue

                entry = _build_entry(
                    entry_id=f"{csv_path.stem}:{index + 1}",
                    name=str(row.get("name") or "").strip(),
                    smiles=smiles,
                    label=row.get("label"),
                    source_dataset=csv_path.stem,
                    notes=str(row.get("notes") or "").strip(),
                    tox_class=[],
                )
                if entry is not None:
                    rows.append(entry)

    return rows


@lru_cache(maxsize=1)
def _load_retrieval_database() -> Dict[str, Any]:
    firestore_rows = _load_firestore_database()
    if firestore_rows:
        return {"rows": firestore_rows, "source": "firestore"}

    return {"rows": _load_csv_database(), "source": "csv_fallback"}

def retrieve_similar_molecules(
    smiles: str,
    *,
    top_k: int = 5,
    min_similarity: float = 0.15,
) -> Dict[str, Any]:
    validated = canonicalize_smiles(smiles)
    if not validated.get("valid"):
        return {
            "query_smiles": smiles,
            "canonical_smiles": None,
            "matches": [],
            "error": validated.get("error") or "invalid_smiles",
            "db_size": 0,
        }

    canonical_smiles = str(validated["canonical_smiles"])
    query_fp = fingerprint_from_smiles(canonical_smiles)
    database_payload = _load_retrieval_database()
    database = database_payload.get("rows", []) if isinstance(database_payload, dict) else []
    actual_source = database_payload.get("source") if isinstance(database_payload, dict) else None
    firestore_state = get_firestore_availability()
    db_source = str(actual_source or ("firestore" if firestore_state.get("ready") else "csv_fallback"))
    if not database:
        return {
            "query_smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "matches": [],
            "error": "retrieval_database_empty",
            "db_size": 0,
            "db_source": db_source,
            "firestore": firestore_state,
        }

    if query_fp is None:
        return {
            "query_smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "matches": [],
            "error": "fingerprint_generation_failed",
            "db_size": len(database),
            "db_source": db_source,
            "firestore": firestore_state,
        }

    matches: List[Dict[str, Any]] = []
    for row in database:
        similarity = tanimoto_similarity(query_fp, row.get("fingerprint"))
        if similarity is None or similarity < float(min_similarity):
            continue

        matches.append(
            {
                "entry_id": row["entry_id"],
                "name": row["name"],
                "smiles": row["smiles"],
                "canonical_smiles": row["canonical_smiles"],
                "similarity": round(float(similarity), 4),
                "label": row["label"],
                "source": row["source_dataset"],
                "notes": row["notes"],
                "tox_class": row.get("tox_class", []),
                "is_exact_match": row["canonical_smiles"] == canonical_smiles,
            }
        )

    matches.sort(
        key=lambda item: (
            float(item.get("similarity", 0.0)),
            1 if item.get("label") == "Toxic" else 0,
        ),
        reverse=True,
    )

    return {
        "query_smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "matches": matches[: max(int(top_k), 0)],
        "error": None,
        "db_size": len(database),
        "db_source": db_source,
        "firestore": firestore_state,
    }
