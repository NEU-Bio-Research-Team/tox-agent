"""Attribution service.

Kept apart from prediction on purpose (plan section 3.2): attribution costs a
backward pass, so it is its own request rather than a field that quietly slows
every prediction down.

Two rules from the plan:

* numeric importance only — no matplotlib, no base64 image. Nothing here
  imports a plotting library, so the runtime image does not need one.
* a timeout is a typed partial failure. It never alters, and never blocks, the
  prediction itself.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from ..domain.endpoints import TOX21_TASK_INDEX, Endpoint
from ..scientific.artifacts import ArtifactError
from ..scientific.featurization.rdkit_resolver import resolve
from ..scientific.registry import ModelRegistry

DEFAULT_TIMEOUT_MS = 30_000


@dataclass(frozen=True)
class AttributionService:
    registry: ModelRegistry
    timeout_ms: int = DEFAULT_TIMEOUT_MS

    def attribute(
        self, smiles: str, endpoint: str, task: str | None = None,
        method: str = "grad_x_input", model_id: str | None = None,
    ) -> dict[str, Any]:
        """``model_id`` pins which admitted model explains this endpoint.

        Without it the caller got whichever model `for_capability` happened to
        resolve, so a prediction made by model B could be handed an
        explanation computed by model A the moment an endpoint had two
        admitted models (I11). The registry already refuses an ambiguous
        capability; this is how a caller answers it.
        """
        endpoint_enum = Endpoint(endpoint)
        if endpoint_enum is Endpoint.TOX21 and task is None:
            raise ValueError(
                "attributing the tox21 endpoint requires a task; the twelve assays are "
                "independent and a combined attribution would not mean anything"
            )
        if endpoint_enum is not Endpoint.TOX21 and task is not None:
            raise ValueError(f"task is only meaningful for tox21, not {endpoint}")

        molecule = resolve(smiles)
        provider = self.registry.resolve(capability=endpoint_enum.value, model_id=model_id)
        attribute = getattr(provider, "token_attribution", None)
        if attribute is None:
            raise ArtifactError(
                f"[{provider.model_id}] does not implement attribution for {endpoint}"
            )

        started = time.perf_counter()
        try:
            raw = attribute(
                molecule.canonical_smiles,
                head=endpoint_enum.value,
                task_index=TOX21_TASK_INDEX[task] if task else None,
                method=method,
            )
        except Exception as exc:  # noqa: BLE001 — reported, never silently dropped
            return {
                "status": "failed",
                "error": type(exc).__name__,
                "message": str(exc),
                "input_smiles": smiles,
                "canonical_smiles": molecule.canonical_smiles,
                "endpoint": endpoint,
                "task": task,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        duration_ms = (time.perf_counter() - started) * 1000

        status = "completed"
        note = None
        if duration_ms > self.timeout_ms:
            status = "partial"
            note = (
                f"attribution took {duration_ms:.0f} ms, over the {self.timeout_ms} ms "
                "budget; scores are returned but should be treated as best-effort"
            )

        return {
            "status": status,
            "input_smiles": smiles,
            "canonical_smiles": molecule.canonical_smiles,
            "endpoint": endpoint,
            "task": task,
            "probability": raw["probability"],
            "tokens": raw["tokens"],
            "metadata": {
                "method": raw["method"],
                "target": raw.get("target", "logit"),
                "mapping_version": raw.get("mapping_version"),
                "model_id": raw["model_id"],
                "deterministic": True,
                "duration_ms": round(duration_ms, 2),
                "timeout_ms": self.timeout_ms,
                "note": note,
            },
        }
