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
def get_firestore_client() -> Optional[Any]:
    """Return a Firestore client, or None when Firestore is disabled/unavailable."""
    if not _is_enabled(os.getenv("MOLRAG_FIRESTORE_ENABLED", "true")):
        return None

    if firebase_admin is None or firestore is None:
        return None

    try:
        if not firebase_admin._apps:
            service_account = _resolve_service_account_path()
            project_id = _clean_env(os.getenv("FIRESTORE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"))
            options: Dict[str, str] = {}
            if project_id:
                options["projectId"] = project_id

            if service_account is not None and credentials is not None:
                cred = credentials.Certificate(str(service_account))
                firebase_admin.initialize_app(cred, options if options else None)
            else:
                firebase_admin.initialize_app(options=options if options else None)

        database_id = _clean_env(os.getenv("FIRESTORE_DATABASE_ID"))
        if database_id and database_id != "(default)":
            try:
                return firestore.client(database_id=database_id)
            except TypeError:
                return firestore.client()

        return firestore.client()
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_firestore_availability() -> Dict[str, Any]:
    """Return a diagnostic snapshot of Firestore availability."""
    if not _is_enabled(os.getenv("MOLRAG_FIRESTORE_ENABLED", "true")):
        return {
            "enabled": False,
            "ready": False,
            "reason": "disabled_by_env",
        }

    service_account = _resolve_service_account_path()
    client = get_firestore_client()

    probe = _probe_firestore_client(client) if client is not None else {"ready": False, "reason": "client_unavailable"}

    return {
        "enabled": True,
        "ready": bool(client is not None and probe.get("ready")),
        "reason": probe.get("reason"),
        "service_account": str(service_account) if service_account is not None else None,
        "database_id": _clean_env(os.getenv("FIRESTORE_DATABASE_ID")) or "(default)",
    }


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
