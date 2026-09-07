"""Default registry assembly.

One place that knows which provider implementation backs which manifest entry.
Optional models that cannot load leave a recorded reason in `registry.errors`
rather than failing startup — a required one still fails loud.
"""
from __future__ import annotations

from pathlib import Path

from .artifacts import ArtifactSpec
from .providers.clintox_smilesgnn import make_factory as clintox_factory
from .providers.herg_tox21_chemberta import factory as chemberta_factory
from .registry import ModelRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "artifacts" / "predictor-manifest.yaml"
DEFAULT_CLINTOX_CONFIG = REPO_ROOT / "config" / "smilesgnn_config.yaml"


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
        eager_load=eager_load,
    )
