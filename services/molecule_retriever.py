from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from .fingerprint_service import canonicalize_smiles, fingerprint_from_smiles, tanimoto_similarity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RETRIEVAL_FILES = [
    PROJECT_ROOT / "test_data" / "reference_panel.csv",
    PROJECT_ROOT / "test_data" / "screening_library.csv",
]


def _normalize_label(raw_label: Any) -> str:
    text = str(raw_label or "").strip().lower()
    if text in {"1", "1.0", "true", "toxic", "positive", "high"}:
        return "Toxic"
    if text in {"0", "0.0", "false", "safe", "non-toxic", "negative", "low"}:
        return "Non-toxic"
    return str(raw_label or "Unknown").strip() or "Unknown"


@lru_cache(maxsize=1)
def _load_retrieval_database() -> List[Dict[str, Any]]:
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

                validated = canonicalize_smiles(smiles)
                if not validated.get("valid"):
                    continue

                canonical_smiles = str(validated["canonical_smiles"])
                fingerprint = fingerprint_from_smiles(canonical_smiles)
                if fingerprint is None:
                    continue

                rows.append(
                    {
                        "entry_id": f"{csv_path.stem}:{index + 1}",
                        "name": str(row.get("name") or canonical_smiles),
                        "smiles": smiles,
                        "canonical_smiles": canonical_smiles,
                        "label": _normalize_label(row.get("label")),
                        "source_dataset": csv_path.stem,
                        "notes": str(row.get("notes") or "").strip(),
                        "fingerprint": fingerprint,
                    }
                )

    return rows


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
    database = _load_retrieval_database()
    if not database:
        return {
            "query_smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "matches": [],
            "error": "retrieval_database_empty",
            "db_size": 0,
        }

    if query_fp is None:
        return {
            "query_smiles": smiles,
            "canonical_smiles": canonical_smiles,
            "matches": [],
            "error": "fingerprint_generation_failed",
            "db_size": len(database),
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
    }
