"""Default registry assembly.

One place that knows which provider implementation backs which manifest entry.
Optional models that cannot load leave a recorded reason in `registry.errors`
rather than failing startup — a required one still fails loud.
"""
from __future__ import annotations

import os
from pathlib import Path

from .artifacts import ArtifactSpec
from .providers.clintox_smilesgnn import make_factory as clintox_factory
from .providers.herg_tox21_chemberta import factory as chemberta_factory
from .registry import ModelRegistry

# In development the package lives at ``predictor/src/toxpred``; release
# images intentionally copy only ``toxpred`` into ``/app``. Derive both roots
# from the observed source layout instead of indexing an assumed parent chain.
SOURCE_ROOT = Path(__file__).resolve().parents[2]
PREDICTOR_ROOT = SOURCE_ROOT.parent if SOURCE_ROOT.name == "src" else SOURCE_ROOT
WORKSPACE_ROOT = PREDICTOR_ROOT.parents[1] if PREDICTOR_ROOT.parent.name == "backend" else PREDICTOR_ROOT
DEFAULT_MANIFEST = PREDICTOR_ROOT / "registry" / "predictor-manifest.yaml"
DEFAULT_CLINTOX_CONFIG = PREDICTOR_ROOT / "configs" / "smilesgnn_config.yaml"


def _clintox(spec: ArtifactSpec, *, device: str = "cpu"):
    config = spec.model_config_path or DEFAULT_CLINTOX_CONFIG
    return clintox_factory(config, device=device)(spec)


PROVIDER_FACTORIES = {
    "herg_tox21_chemberta": chemberta_factory,
    "clintox_smilesgnn": _clintox,
}


def provider_factories(device: str) -> dict[str, object]:
    """Bind the explicitly selected device into every provider factory."""
    return {
        "herg_tox21_chemberta": lambda spec: chemberta_factory(spec, device=device),
        "clintox_smilesgnn": lambda spec: _clintox(spec, device=device),
    }


def resolve_manifest(manifest_path: Path | None = None) -> Path:
    """The manifest this install will actually read, or a usable error.

    A missing manifest used to surface as a bare ``FileNotFoundError`` naming
    a path the operator had never configured, which is how I23 stayed hidden:
    the wrong default looked like a missing artifact. Say which path was
    resolved, how it was chosen, and what would change it.
    """
    resolved = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST
    if not resolved.is_file():
        source = "TOXPRED_MANIFEST" if manifest_path else "this install's registry directory"
        raise FileNotFoundError(
            f"predictor manifest not found at {resolved} (chosen from {source}). "
            "Provision the model registry, or set TOXPRED_MANIFEST to an "
            "existing manifest."
        )
    return resolved


def build_registry(
    manifest_path: Path | None = None, *, eager_load: bool = True, device: str = "cpu"
) -> ModelRegistry:
    return ModelRegistry.from_manifest(
        resolve_manifest(manifest_path),
        provider_factories(device),
        models_root=Path(os.environ["MODELS_ROOT"]) if os.environ.get("MODELS_ROOT") else None,
        eager_load=eager_load,
    )
