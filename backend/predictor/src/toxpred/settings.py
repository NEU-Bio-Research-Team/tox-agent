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
    #: ``None`` means "wherever this install keeps its registry" — resolved by
    #: scientific/bootstrap.py, which already derives the predictor root from
    #: the observed source layout and therefore works in a checkout, a wheel
    #: and the image alike. Naming a default here instead (I23) produced
    #: ``src/artifacts/predictor-manifest.yaml``, a path no layout has, and
    #: silently shadowed the correct default because it was never ``None``.
    manifest_path: Path | None
    device: str
    max_batch_size: int
    eager_load: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            manifest_path=Path(_manifest) if (_manifest := (os.getenv("TOXPRED_MANIFEST") or "").strip()) else None,
            device=os.getenv("TOXPRED_DEVICE", "cpu").strip() or "cpu",
            max_batch_size=_int("TOXPRED_MAX_BATCH_SIZE", 256),
            eager_load=os.getenv("TOXPRED_EAGER_LOAD", "1").strip().lower()
            not in {"0", "false", "no"},
        )
