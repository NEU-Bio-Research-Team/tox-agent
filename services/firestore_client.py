from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except Exception:
    firebase_admin = None
    credentials = None
    firestore = None


_FALSE_VALUES = {"0", "false", "no", "off", "disabled", "none"}


def _clean_env(value: Optional[str]) -> str:
    return str(value or "").strip()


def _is_enabled(raw_value: Optional[str]) -> bool:
    value = _clean_env(raw_value).lower()
    if value == "":
        return True
    return value not in _FALSE_VALUES


def _resolve_service_account_path() -> Optional[Path]:
    env_candidates = [
        os.getenv("FIREBASE_SERVICE_ACCOUNT"),
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    ]

    for raw_path in env_candidates:
        candidate = Path(_clean_env(raw_path)).expanduser() if raw_path else None
        if candidate and candidate.exists():
            return candidate

    local_candidate = Path("serviceAccountKey.json")
    if local_candidate.exists():
        return local_candidate.resolve()

    return None


def _resolve_project_id() -> Optional[str]:
    project_id = _clean_env(os.getenv("FIRESTORE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"))
    return project_id or None


def _resolve_credential_source(service_account: Optional[Path]) -> str:
    if service_account is not None:
        return "service_account_file"
    if _clean_env(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")):
        return "application_default_credentials_env"
    return "application_default_credentials"


def _create_firestore_client(database_id: str) -> Any:
    if database_id and database_id != "(default)":
        try:
            return firestore.client(database_id=database_id)
        except TypeError:
            return firestore.client()
    return firestore.client()


def _probe_firestore_client(client: Any) -> Dict[str, Any]:
    """Run a lightweight probe query to confirm the configured database is reachable."""
    try:
        stream = client.collection("_molrag_probe").limit(1).stream()
        next(stream, None)
        return {
            "ready": True,
            "reason": None,
        }
    except Exception as exc:
        return {
            "ready": False,
            "reason": f"{type(exc).__name__}: {str(exc)[:180]}",
        }


@lru_cache(maxsize=1)
def _resolve_firestore_state() -> Dict[str, Any]:
    """Resolve a working Firestore client and expose diagnostics for the chosen database."""
    service_account = _resolve_service_account_path()
    project_id = _resolve_project_id()
    configured_database_id = _clean_env(os.getenv("FIRESTORE_DATABASE_ID")) or "(default)"
    base_state: Dict[str, Any] = {
        "enabled": True,
        "ready": False,
        "reason": None,
        "service_account": str(service_account) if service_account is not None else None,
        "credential_source": _resolve_credential_source(service_account),
        "project_id": project_id,
        "configured_database_id": configured_database_id,
        "database_id": configured_database_id,
        "used_database_fallback": False,
        "fallback_reason": None,
        "attempts": [],
        "client": None,
    }

    if not _is_enabled(os.getenv("MOLRAG_FIRESTORE_ENABLED", "true")):
        base_state.update({
            "enabled": False,
            "reason": "disabled_by_env",
        })
        return base_state

    if firebase_admin is None or firestore is None:
        base_state.update({
            "reason": "firebase_admin_unavailable",
        })
        return base_state

    try:
        if not firebase_admin._apps:
            options: Dict[str, str] = {}
            if project_id:
                options["projectId"] = project_id

            if service_account is not None and credentials is not None:
                cred = credentials.Certificate(str(service_account))
                firebase_admin.initialize_app(cred, options if options else None)
            else:
                firebase_admin.initialize_app(options=options if options else None)
    except Exception as exc:
        base_state.update({
            "reason": f"initialize_app_failed: {type(exc).__name__}: {str(exc)[:180]}",
        })
        return base_state

    candidate_database_ids = [configured_database_id]
    if configured_database_id != "(default)":
        candidate_database_ids.append("(default)")

    fallback_reason: Optional[str] = None
    attempts: List[Dict[str, Any]] = []
    for database_id in candidate_database_ids:
        try:
            client = _create_firestore_client(database_id)
        except Exception as exc:
            reason = f"create_client_failed: {type(exc).__name__}: {str(exc)[:180]}"
            attempts.append({
                "database_id": database_id,
                "ready": False,
                "reason": reason,
            })
            if database_id == configured_database_id:
                fallback_reason = reason
            continue

        probe = _probe_firestore_client(client)
        attempt = {
            "database_id": database_id,
            "ready": probe.get("ready", False),
            "reason": probe.get("reason"),
        }
        attempts.append(attempt)

        if probe.get("ready"):
            used_database_fallback = database_id != configured_database_id
            base_state.update({
                "ready": True,
                "reason": None if not used_database_fallback else "configured_database_unavailable_fell_back_to_default",
                "database_id": database_id,
                "used_database_fallback": used_database_fallback,
                "fallback_reason": fallback_reason,
                "attempts": attempts,
                "client": client,
            })
            return base_state

        if database_id == configured_database_id:
            fallback_reason = probe.get("reason")

    base_state.update({
        "reason": fallback_reason or (attempts[-1].get("reason") if attempts else "client_unavailable"),
        "fallback_reason": fallback_reason,
        "attempts": attempts,
    })
    return base_state


@lru_cache(maxsize=1)
def get_firestore_client() -> Optional[Any]:
    """Return a Firestore client, or None when Firestore is disabled/unavailable."""
    return _resolve_firestore_state().get("client")


@lru_cache(maxsize=1)
def get_firestore_availability() -> Dict[str, Any]:
    """Return a diagnostic snapshot of Firestore availability."""
    state = dict(_resolve_firestore_state())
    state.pop("client", None)
    return state


def fetch_collection_documents(collection_name: str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch collection documents as dictionaries. Returns [] when unavailable."""
    client = get_firestore_client()
    if client is None:
        return []

    try:
        ref = client.collection(collection_name)
        query = ref.limit(int(limit)) if limit is not None else ref
        documents = query.stream()
        rows: List[Dict[str, Any]] = []
        for snapshot in documents:
            payload = snapshot.to_dict() or {}
            payload.setdefault("doc_id", snapshot.id)
            rows.append(payload)
        return rows
    except Exception:
        return []
