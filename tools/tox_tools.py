from __future__ import annotations

import os
import threading
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


DEFAULT_MODEL_SERVER_PORT = (
    os.getenv("AIP_HTTP_PORT")
    or os.getenv("PORT")
    or "8000"
).strip()
_AWS_URL = (os.getenv("MODEL_SERVER_URL_AWS") or "").rstrip("/")
_GCP_URL = (os.getenv("MODEL_SERVER_URL_GCP") or "").rstrip("/")
_LEGACY_URL = os.getenv(
    "MODEL_SERVER_URL",
    f"http://127.0.0.1:{DEFAULT_MODEL_SERVER_PORT}",
).rstrip("/")
MODEL_SERVER_TIMEOUT = _get_env_float("MODEL_SERVER_TIMEOUT", 240.0)
MODEL_SERVER_HEALTH_TIMEOUT = _get_env_float("MODEL_SERVER_HEALTH_TIMEOUT", 12.0)
BATCH_TIMEOUT = max(MODEL_SERVER_TIMEOUT * 4.0, 120.0)


class ModelServerRouter:
    """Route model-server requests to AWS first, then fall back to GCP."""

    def __init__(self) -> None:
        urls: List[str] = []
        if _AWS_URL:
            urls.append(_AWS_URL)
        if _GCP_URL:
            urls.append(_GCP_URL)
        if not urls:
            urls.append(_LEGACY_URL)

        self._urls = urls
        self._active_url = urls[0]
        self._last_check = 0.0
        self._check_interval = 60.0
        self._lock = threading.Lock()

    def _probe(self, base_url: str, timeout: float | None = None) -> bool:
        try:
            response = httpx.get(
                f"{base_url}/health",
                timeout=timeout or MODEL_SERVER_HEALTH_TIMEOUT,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _resolve_url(self) -> str:
        now = time.monotonic()
        with self._lock:
            active_url = self._active_url
            should_recheck_primary = (
                len(self._urls) > 1
                and active_url != self._urls[0]
                and (now - self._last_check) > self._check_interval
            )

        if should_recheck_primary and self._probe(self._urls[0]):
            with self._lock:
                self._active_url = self._urls[0]
                self._last_check = now
                active_url = self._active_url

        return active_url

    def _record_success(self, url: str) -> None:
        with self._lock:
            if url != self._active_url:
                self._active_url = url
            self._last_check = time.monotonic()

    def _record_failure(self) -> None:
        with self._lock:
            self._last_check = time.monotonic()

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        json: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        active_url = self._resolve_url()
        urls_to_try = [active_url] + [url for url in self._urls if url != active_url]
        last_exc: Exception | None = None

        for url in urls_to_try:
            try:
                response = httpx.request(
                    method,
                    f"{url}{path}",
                    json=json,
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError("invalid_response")
                self._record_success(url)
                return data
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code is not None and status_code < 500 and status_code != 429:
                    raise
                self._record_failure()
                continue
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RequestError) as exc:
                last_exc = exc
                self._record_failure()
                continue
            except Exception as exc:
                last_exc = exc
                break

        raise last_exc or RuntimeError("all_model_servers_failed")

    def get(self, path: str, timeout: float) -> Dict[str, Any]:
        return self._request("GET", path, timeout=timeout)

    def post(self, path: str, json: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        return self._request("POST", path, json=json, timeout=timeout)

    def active_backend(self) -> str:
        url = self._resolve_url()
        if _AWS_URL and url == _AWS_URL:
            return "aws"
        if _GCP_URL and url == _GCP_URL:
            return "gcp"
        return "legacy"


_router = ModelServerRouter()


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
    inference_backend: str = "xsmiles",
    binary_tox_model: str = "pretrained_2head_herg_chemberta_model",
    tox_type_model: str = "tox21_ensemble_3_best",
) -> Dict[str, Any]:
    """Run full model-server toxicity analysis for one validated SMILES.

    This tool calls the routed ``POST /analyze`` model-server endpoint and returns the unified
    clinical/mechanistic/explainer payload. Prefer canonical SMILES returned by
    ``validate_smiles``.

    Args:
        smiles: Valid SMILES string (ideally canonical).
        clinical_threshold: Toxicity threshold for clinical binary decision.
        mechanism_threshold: Default mechanism threshold when task-specific
            thresholds are unavailable.
        inference_backend: Backend selector for model inference.

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

    payload = {
        "smiles": smiles,
        "clinical_threshold": float(clinical_threshold),
        "mechanism_threshold": float(mechanism_threshold),
        "inference_backend": str(inference_backend),
        "explain_only_if_alert": False,
        "binary_tox_model": str(binary_tox_model),
        "tox_type_model": str(tox_type_model),
    }

    try:
        data = _router.post("/analyze", json=payload, timeout=MODEL_SERVER_TIMEOUT)
        data.setdefault("error", None)
        data["_backend"] = _router.active_backend()
        return data
    except httpx.TimeoutException:
        return {
            "error": f"all_backends_timeout_{MODEL_SERVER_TIMEOUT}s",
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

    This tool calls the routed ``POST /predict/batch`` endpoint and is intended
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
        payload = _router.post(
            "/predict/batch",
            json={"smiles_list": smiles_list},
            timeout=BATCH_TIMEOUT,
        )

        results = payload.get("results", [])
        total = int(payload.get("total", len(smiles_list)) or len(smiles_list))
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

    timeouts = [MODEL_SERVER_HEALTH_TIMEOUT, max(MODEL_SERVER_HEALTH_TIMEOUT * 2.0, 20.0)]
    for idx, timeout in enumerate(timeouts):
        try:
            data = _router.get("/health", timeout=timeout)
            latency_ms = (time.perf_counter() - start) * 1000.0
            status = data.get("status") if isinstance(data, dict) else None
            payload = {
                "healthy": True,
                "status": status or "ok",
                "latency_ms": latency_ms,
                "error": None,
                "backend": _router.active_backend(),
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
