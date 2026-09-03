"""Runtime settings.

The one place that reads the environment. Every other module receives resolved
values, so no module can invent its own default halfway down a call stack —
the failure mode that left five different clinical thresholds in the code this
replaces.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    manifest_path: Path
    device: str
    max_batch_size: int
    eager_load: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            manifest_path=Path(
                os.getenv("TOXPRED_MANIFEST") or REPO_ROOT / "artifacts" / "manifest.yaml"
            ),
            device=os.getenv("TOXPRED_DEVICE", "cpu").strip() or "cpu",
            max_batch_size=_int("TOXPRED_MAX_BATCH_SIZE", 256),
            eager_load=os.getenv("TOXPRED_EAGER_LOAD", "1").strip().lower()
            not in {"0", "false", "no"},
        )
