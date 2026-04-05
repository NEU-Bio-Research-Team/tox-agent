"""Workspace mode helpers for enabling/disabling dataset workflows."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_MODE_PATH = PROJECT_ROOT / "config" / "workspace_mode.yaml"

_DEFAULT_MODE = {
    "mode": "tox21_only",
    "primary_dataset": "tox21",
    "clintox_enabled": False,
    "tox21_enabled": True,
    "threshold_policy": "balanced",
    "clinical_threshold": None,
    "message": "ClinTox workflows are disabled in this workspace mode.",
}

_THRESHOLD_POLICY_DEFAULTS = {
    "conservative": 0.5,
    "balanced": 0.35,
    "safety_first": 0.3,
}


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def get_workspace_mode() -> Dict[str, object]:
    """Load workspace mode settings from config/workspace_mode.yaml."""
    mode = dict(_DEFAULT_MODE)

    if WORKSPACE_MODE_PATH.exists():
        with open(WORKSPACE_MODE_PATH, "r") as f:
            raw_cfg = yaml.safe_load(f) or {}
        raw_workspace = raw_cfg.get("workspace", {})

        mode["mode"] = raw_workspace.get("mode", mode["mode"])
        mode["primary_dataset"] = raw_workspace.get(
            "primary_dataset", mode["primary_dataset"]
        )
        mode["clintox_enabled"] = bool(
            raw_workspace.get("clintox_enabled", mode["clintox_enabled"])
        )
        mode["tox21_enabled"] = bool(
            raw_workspace.get("tox21_enabled", mode["tox21_enabled"])
        )
        mode["threshold_policy"] = str(
            raw_workspace.get("threshold_policy", mode["threshold_policy"])
        ).strip().lower()
        mode["clinical_threshold"] = _safe_float(
            raw_workspace.get("clinical_threshold", mode["clinical_threshold"])
        )
        mode["message"] = raw_workspace.get("message", mode["message"])

    return mode


def resolve_default_clinical_threshold() -> float:
    """Resolve clinical threshold with env override then workspace policy."""
    env_threshold = _safe_float(os.getenv("CLINICAL_THRESHOLD"))
    if env_threshold is not None:
        return env_threshold

    mode = get_workspace_mode()
    explicit_threshold = _safe_float(mode.get("clinical_threshold"))
    if explicit_threshold is not None:
        return explicit_threshold

    policy = str(mode.get("threshold_policy", "balanced")).strip().lower()
    return float(_THRESHOLD_POLICY_DEFAULTS.get(policy, _THRESHOLD_POLICY_DEFAULTS["balanced"]))


def get_threshold_policy() -> str:
    mode = get_workspace_mode()
    return str(mode.get("threshold_policy", "balanced")).strip().lower()


def is_clintox_enabled() -> bool:
    """Return whether ClinTox workflows are enabled."""
    return bool(get_workspace_mode()["clintox_enabled"])


def assert_clintox_enabled(context: str) -> None:
    """Raise an actionable error if ClinTox workflows are disabled."""
    if is_clintox_enabled():
        return

    mode = get_workspace_mode()
    message = mode.get("message", _DEFAULT_MODE["message"])
    raise RuntimeError(
        f"[DISABLED:CLINTOX] {context} is disabled. {message} "
        f"Active workspace mode: '{mode.get('mode')}'. "
        "Use Tox21 scripts (e.g., scripts/train_tox21_gatv2.py) instead."
    )


def assert_tox21_enabled(context: str) -> None:
    """Raise an actionable error if Tox21 workflows are disabled."""
    mode = get_workspace_mode()
    if bool(mode["tox21_enabled"]):
        return

    raise RuntimeError(
        f"[DISABLED:TOX21] {context} is disabled in workspace mode "
        f"'{mode.get('mode')}'."
    )
