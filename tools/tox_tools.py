from __future__ import annotations

import os
import time
import inspect
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
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


DEFAULT_MODEL_SERVER_PORT = (
    os.getenv("AIP_HTTP_PORT")
    or os.getenv("PORT")
    or "8000"
).strip()
MODEL_SERVER_URL = os.getenv(
    "MODEL_SERVER_URL",
    f"http://127.0.0.1:{DEFAULT_MODEL_SERVER_PORT}",
).rstrip("/")
MODEL_SERVER_TIMEOUT = _get_env_float("MODEL_SERVER_TIMEOUT", 30.0)
MODEL_SERVER_HEALTH_TIMEOUT = _get_env_float("MODEL_SERVER_HEALTH_TIMEOUT", 12.0)
BATCH_TIMEOUT = max(MODEL_SERVER_TIMEOUT * 4.0, 120.0)


def _is_local_model_server_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    hostname = (parsed.hostname or "").lower()
    return hostname in {"127.0.0.1", "localhost", "::1"}


def _should_use_direct_model_server_call() -> bool:
    override = os.getenv("TOX_AGENT_DIRECT_ANALYZE")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "no", "off"}
    # Prefer the in-process route by default to avoid self-call loops when the
    # app's own public URL is stored in MODEL_SERVER_URL.
    return True


def validate_smiles(smiles: str) -> Dict[str, Any]:
    """Validate one SMILES string using RDKit and return canonical form.

    Use this tool first before running model inference. It confirms whether a
    string can be parsed as a molecule and provides canonical SMILES for stable
    downstream calls.

    Args:
        smiles: Raw SMILES input from user/session state.

    Returns:
        A dict with keys:
        - valid (bool): True if RDKit parses the SMILES.
        - canonical_smiles (str | None): Canonicalized SMILES when valid.
        - error (str | None): Parse/validation error when invalid.
        - atom_count (int | None): Number of atoms in parsed molecule.
    """
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


def analyze_molecule(
    smiles: str,
    clinical_threshold: float = 0.35,
    mechanism_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run full model-server toxicity analysis for one validated SMILES.

    This tool calls ``POST {MODEL_SERVER_URL}/analyze`` and returns the unified
    clinical/mechanistic/explainer payload. Prefer canonical SMILES returned by
    ``validate_smiles``.

    Args:
        smiles: Valid SMILES string (ideally canonical).
        clinical_threshold: Toxicity threshold for clinical binary decision.
        mechanism_threshold: Default mechanism threshold when task-specific
            thresholds are unavailable.

    Returns:
        Dict from model server containing keys such as ``clinical``,
        ``mechanism``, ``explanation``, ``final_verdict`` and ``error``.
        On transport/server failure, returns ``error`` and
        ``final_verdict=ANALYSIS_FAILED``.
    """
    if not smiles or not smiles.strip():
        return {
            "error": "smiles_empty",
            "smiles": smiles,
            "final_verdict": "ANALYSIS_FAILED",
        }

    if _should_use_direct_model_server_call():
        try:
            from model_server.main import analyze as analyze_route
            from model_server.schemas import AnalyzeRequest
            import asyncio

            request = AnalyzeRequest(
                smiles=smiles,
                clinical_threshold=float(clinical_threshold),
                mechanism_threshold=float(mechanism_threshold),
                explain_only_if_alert=False,
            )

            if inspect.iscoroutinefunction(analyze_route):
                def _run_async_route() -> Any:
                    return asyncio.run(analyze_route(request))

                with ThreadPoolExecutor(max_workers=1) as executor:
                    response = executor.submit(_run_async_route).result(timeout=MODEL_SERVER_TIMEOUT)
            else:
                response = analyze_route(request)

            if hasattr(response, "model_dump"):
                data = response.model_dump()
            elif isinstance(response, dict):
                data = response
            else:
                data = None

            if isinstance(data, dict):
                data.setdefault("error", None)
                return data
        except Exception:
            # Fall back to HTTP below if the in-process path is unavailable.
            pass

    try:
        response = httpx.post(
            f"{MODEL_SERVER_URL}/analyze",
            json={
                "smiles": smiles,
                "clinical_threshold": float(clinical_threshold),
                "mechanism_threshold": float(mechanism_threshold),
                "explain_only_if_alert": False,
            },
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
    """Run batch clinical predictions for multiple SMILES strings.

    This tool calls ``POST {MODEL_SERVER_URL}/predict/batch`` and is intended
    for throughput-oriented screening. It does not return full mechanism/
    explainer outputs like ``analyze_molecule``.

    Args:
        smiles_list: List of SMILES strings. Maximum length is 50.

    Returns:
        Dict with keys:
        - results (list): Per-molecule prediction payloads.
        - total (int): Number of submitted molecules.
        - success_count (int): Count of non-failed predictions.
        - error (str | None): Validation/request error.
    """
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
        payload = response.json()

        # Support both legacy list payloads and current BatchPredictResponse dict.
        if isinstance(payload, dict):
            results = payload.get("results", [])
            total = int(payload.get("total", len(smiles_list)) or len(smiles_list))
        elif isinstance(payload, list):
            results = payload
            total = len(results)
        else:
            return {
                "error": "invalid_response",
                "results": [],
                "total": len(smiles_list),
                "success_count": 0,
            }

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
            and item.get("label") not in {"PARSE_ERROR", "UNKNOWN"}
        )
        return {
            "results": results,
            "total": total,
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
    """Check model server availability and latency.

    Use this before other server-dependent tools to gate execution and provide
    actionable error messages when backend services are unreachable.

    Returns:
        Dict with keys:
        - healthy (bool): True if health endpoint responds successfully.
        - status (str): Health status string from backend.
        - latency_ms (float): End-to-end request latency.
        - error (str | None): Connectivity/HTTP error when unhealthy.
    """
    start = time.perf_counter()
    last_error: Exception | None = None

    # Internal self-calls on Cloud Run can be bursty under load.
    # Retry once with a larger timeout before declaring the server unhealthy.
    timeouts = [MODEL_SERVER_HEALTH_TIMEOUT, max(MODEL_SERVER_HEALTH_TIMEOUT * 2.0, 20.0)]
    for idx, timeout in enumerate(timeouts):
        try:
            response = httpx.get(
                f"{MODEL_SERVER_URL}/health",
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            latency_ms = (time.perf_counter() - start) * 1000.0
            status = data.get("status") if isinstance(data, dict) else None
            payload = {
                "healthy": True,
                "status": status or "ok",
                "latency_ms": latency_ms,
                "error": None,
            }
            if idx > 0:
                payload["retry_count"] = idx
            return payload
        except Exception as exc:
            last_error = exc
            if idx < len(timeouts) - 1:
                time.sleep(0.2)

    latency_ms = (time.perf_counter() - start) * 1000.0
    return {
        "healthy": False,
        "status": "unreachable",
        "latency_ms": latency_ms,
        "error": str(last_error) if last_error is not None else "unknown_error",
    }
