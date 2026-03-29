"""Workspace mode helpers for enabling/disabling dataset workflows."""

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
    "message": "ClinTox workflows are disabled in this workspace mode.",
}


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
        mode["message"] = raw_workspace.get("message", mode["message"])

    return mode


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
