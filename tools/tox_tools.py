from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import httpx

try:
    from rdkit import Chem
except Exception:
    Chem = None


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8000").rstrip("/")
MODEL_SERVER_TIMEOUT = _get_env_float("MODEL_SERVER_TIMEOUT", 30.0)
MODEL_SERVER_HEALTH_TIMEOUT = _get_env_float("MODEL_SERVER_HEALTH_TIMEOUT", 5.0)
BATCH_TIMEOUT = max(MODEL_SERVER_TIMEOUT * 4.0, 120.0)


def validate_smiles(smiles: str) -> Dict[str, Any]:
    """Validate SMILES locally using RDKit when available."""
    if not smiles or not smiles.strip():
        return {
            "valid": False,
            "canonical_smiles": None,
            "error": "smiles_empty",
            "atom_count": None,
        }

    if Chem is None:
        return {
            "valid": False,
            "canonical_smiles": None,
            "error": "rdkit_not_installed",
            "atom_count": None,
        }

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return {
            "valid": False,
            "canonical_smiles": None,
            "error": f"rdkit_parse_failed: {smiles}",
            "atom_count": None,
        }

    canonical = Chem.MolToSmiles(mol)
    return {
        "valid": True,
        "canonical_smiles": canonical,
        "error": None,
        "atom_count": mol.GetNumAtoms(),
    }


def analyze_molecule(smiles: str) -> Dict[str, Any]:
    """Call model_server /analyze for a single SMILES."""
    if not smiles or not smiles.strip():
        return {
            "error": "smiles_empty",
            "smiles": smiles,
            "final_verdict": "ANALYSIS_FAILED",
        }

    try:
        response = httpx.post(
            f"{MODEL_SERVER_URL}/analyze",
            json={"smiles": smiles},
            timeout=MODEL_SERVER_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            data.setdefault("error", None)
            return data
        return {
            "error": "invalid_response",
            "smiles": smiles,
            "final_verdict": "ANALYSIS_FAILED",
        }
    except httpx.TimeoutException:
        return {
            "error": f"model_server_timeout_{MODEL_SERVER_TIMEOUT}s",
            "smiles": smiles,
            "final_verdict": "ANALYSIS_FAILED",
        }
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        detail = exc.response.text if exc.response is not None else ""
        return {
            "error": f"http_{status}: {detail}",
            "smiles": smiles,
            "final_verdict": "ANALYSIS_FAILED",
        }
    except httpx.RequestError as exc:
        return {
            "error": f"request_error: {exc}",
            "smiles": smiles,
            "final_verdict": "ANALYSIS_FAILED",
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "smiles": smiles,
            "final_verdict": "ANALYSIS_FAILED",
        }


def analyze_molecules_batch(smiles_list: List[str]) -> Dict[str, Any]:
    """Call model_server /predict/batch for a list of SMILES."""
    if not smiles_list:
        return {"results": [], "total": 0, "success_count": 0, "error": "empty"}
    if len(smiles_list) > 50:
        return {
            "error": "batch_limit_exceeded",
            "results": [],
            "total": len(smiles_list),
            "success_count": 0,
        }

    try:
        response = httpx.post(
            f"{MODEL_SERVER_URL}/predict/batch",
            json={"smiles_list": smiles_list},
            timeout=BATCH_TIMEOUT,
        )
        response.raise_for_status()
        results = response.json()
        if not isinstance(results, list):
            return {
                "error": "invalid_response",
                "results": [],
                "total": len(smiles_list),
                "success_count": 0,
            }
        success_count = sum(
            1
            for item in results
            if isinstance(item, dict)
            and item.get("final_verdict") not in {"ANALYSIS_FAILED", "PHAN_TICH_THAT_BAI"}
        )
        return {
            "results": results,
            "total": len(smiles_list),
            "success_count": success_count,
            "error": None,
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "results": [],
            "total": len(smiles_list),
            "success_count": 0,
        }


def check_model_server_health() -> Dict[str, Any]:
    """Call model_server /health and report latency."""
    start = time.perf_counter()
    try:
        response = httpx.get(
            f"{MODEL_SERVER_URL}/health",
            timeout=MODEL_SERVER_HEALTH_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.perf_counter() - start) * 1000.0
        status = data.get("status") if isinstance(data, dict) else None
        return {
            "healthy": True,
            "status": status or "ok",
            "latency_ms": latency_ms,
            "error": None,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "healthy": False,
            "status": "unreachable",
            "latency_ms": latency_ms,
            "error": str(exc),
        }
