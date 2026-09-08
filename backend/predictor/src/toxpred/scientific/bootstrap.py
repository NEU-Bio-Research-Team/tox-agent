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

# ``src/toxpred/scientific/bootstrap.py`` lives three levels below the
# predictor package root. Keep release metadata with that package rather than
# relying on the historical repository-root artifacts directory.
PREDICTOR_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PREDICTOR_ROOT.parents[1]
DEFAULT_MANIFEST = PREDICTOR_ROOT / "registry" / "predictor-manifest.yaml"
DEFAULT_CLINTOX_CONFIG = WORKSPACE_ROOT / "config" / "smilesgnn_config.yaml"


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


def build_registry(
    manifest_path: Path | None = None, *, eager_load: bool = True, device: str = "cpu"
) -> ModelRegistry:
    return ModelRegistry.from_manifest(
        manifest_path or DEFAULT_MANIFEST,
        provider_factories(device),
        models_root=Path(os.environ["MODELS_ROOT"]) if os.environ.get("MODELS_ROOT") else None,
        eager_load=eager_load,
    )
