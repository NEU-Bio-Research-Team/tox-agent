"""Model registry.

Registers a provider only when its artifact verifies. There is no fallback: if a
required model is unavailable the registry is not ready and the service must
answer 503, rather than quietly answering with a different model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from .artifacts import ArtifactError, ArtifactSpec, load_manifest


@dataclass(frozen=True)
class ModelHealth:
    model_id: str
    loaded: bool
    capabilities: frozenset[str]
    detail: str = ""


class RawPrediction(Protocol):
    """Provider output: raw probabilities only, never labels.

    Thresholding is the policy layer's job, so a provider cannot bake an
    operating point into its result.
    """


@runtime_checkable
class ModelProvider(Protocol):
    model_id: str
    capabilities: frozenset[str]

    def load(self) -> None: ...
    def health(self) -> ModelHealth: ...
    def predict(self, canonical_smiles: list[str]) -> list[dict[str, Any]]: ...


ProviderFactory = Callable[[ArtifactSpec], ModelProvider]


class ModelRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, ModelProvider] = {}
        self._specs: dict[str, ArtifactSpec] = {}
        self._errors: dict[str, str] = {}
        self._required: dict[str, bool] = {}

    # -- construction ------------------------------------------------------
    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        factories: dict[str, ProviderFactory],
        *,
        models_root: Path | None = None,
        eager_load: bool = True,
    ) -> "ModelRegistry":
        registry = cls()
        specs = load_manifest(Path(manifest_path), models_root)
        for model_id, spec in specs.items():
            registry._required[model_id] = spec.required
            # Keep the spec even if registration fails, so /v1/models can report
            # what a blocked model is and why rather than guessing.
            registry._specs[model_id] = spec
            factory = factories.get(spec.provider)
            if factory is None:
                msg = f"no provider factory registered for {spec.provider!r}"
                if spec.required:
                    raise ArtifactError(f"[{model_id}] {msg}")
                registry._errors[model_id] = msg
                continue
            try:
                spec.verify()
                provider = factory(spec)
                if provider.capabilities != spec.capabilities:
                    raise ArtifactError(
                        f"[{model_id}] provider capabilities {sorted(provider.capabilities)} "
                        f"do not match manifest {sorted(spec.capabilities)}"
                    )
                if eager_load:
                    provider.load()
            except Exception as exc:  # noqa: BLE001 — required models re-raise below
                registry._errors[model_id] = f"{type(exc).__name__}: {exc}"
                if spec.required:
                    raise
                continue
            registry._providers[model_id] = provider
        return registry

    # -- lookup ------------------------------------------------------------
    def get(self, model_id: str) -> ModelProvider:
        try:
            return self._providers[model_id]
        except KeyError:
            detail = self._errors.get(model_id, "not registered")
            raise ArtifactError(f"model {model_id!r} unavailable: {detail}") from None

    def for_capability(self, capability: str) -> ModelProvider:
        matches = [p for p in self._providers.values() if capability in p.capabilities]
        if not matches:
            raise ArtifactError(
                f"no registered model provides capability {capability!r}. "
                f"registered: {self.describe_capabilities()}"
            )
        if len(matches) > 1:
            raise ArtifactError(
                f"capability {capability!r} is ambiguous across "
                f"{[p.model_id for p in matches]}; resolution must be explicit"
            )
        return matches[0]

    def spec(self, model_id: str) -> ArtifactSpec:
        try:
            return self._specs[model_id]
        except KeyError:
            raise ArtifactError(f"no manifest entry for {model_id!r}") from None

    def describe_capabilities(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for provider in self._providers.values():
            for cap in sorted(provider.capabilities):
                out.setdefault(cap, []).append(provider.model_id)
        return out

    # -- readiness ---------------------------------------------------------
    @property
    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    def unavailable(self) -> dict[str, dict[str, Any]]:
        """Declared models that did not load, and whether that blocks readiness."""
        return {
            model_id: {"required": self._required.get(model_id, True), "reason": reason}
            for model_id, reason in self._errors.items()
        }

    def health(self) -> list[ModelHealth]:
        healths = [p.health() for p in self._providers.values()]
        healths.extend(
            ModelHealth(model_id=mid, loaded=False, capabilities=frozenset(), detail=err)
            for mid, err in self._errors.items()
        )
        return healths

    def is_ready(self, required_capabilities: Iterable[str] = ()) -> tuple[bool, list[str]]:
        """Readiness for the given capabilities.

        A declared-but-optional model that failed to load does not make the
        service unready — it only removes the capabilities it would have
        provided, which the capability check below catches if they were asked
        for. Only a required model failing is a startup failure.
        """
        reasons = [
            f"{mid}: {err}"
            for mid, err in self._errors.items()
            if self._required.get(mid, True)
        ]
        reasons += [
            f"{h.model_id}: not loaded ({h.detail})"
            for h in (p.health() for p in self._providers.values())
            if not h.loaded
        ]
        available = set(self.describe_capabilities())
        missing = sorted(set(required_capabilities) - available)
        reasons += [f"capability {c!r} has no provider" for c in missing]
        return (not reasons, reasons)
