"""ToxicityPredictor — the one entry point for prediction.

Responsibilities kept here rather than in an HTTP handler, so benchmarks and
tests exercise exactly the code the API runs:

* resolve and canonicalise input once;
* resolve an immutable policy snapshot from artifact thresholds;
* call each provider once per batch — the dual-head model yields hERG and Tox21
  together, so requesting both endpoints must not run the backbone twice;
* attach provenance to every result.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .. import __version__
from ..domain.endpoints import TOX21_TASK_ORDER_VERSION, TOX21_TASKS, Endpoint
from ..domain.molecule import InvalidSmilesError, Molecule
from ..domain.policy import PredictionPolicySnapshot
from ..domain.prediction import (
    HergPrediction,
    PredictionResult,
    Tox21AssayPrediction,
    Tox21Prediction,
)
from ..scientific.applicability import assess
from ..scientific.artifacts import ArtifactError
from ..scientific.featurization.rdkit_resolver import resolve
from ..scientific.registry import ModelRegistry

MAX_BATCH_SIZE = 256


@dataclass(frozen=True)
class BatchItemError:
    index: int
    input_smiles: str
    error: str
    detail: str


class ToxicityPredictor:
    def __init__(self, registry: ModelRegistry, *, max_batch_size: int = MAX_BATCH_SIZE) -> None:
        self._registry = registry
        self._max_batch_size = int(max_batch_size)

    # -- policy ------------------------------------------------------------
    def _policy(
        self,
        provider: Any,
        herg_override: float | None,
        tox21_override: Mapping[str, float] | None,
    ) -> PredictionPolicySnapshot:
        return PredictionPolicySnapshot.from_artifact(
            herg_threshold=provider.artifact_herg_threshold,
            tox21_thresholds=provider.artifact_tox21_thresholds,
            herg_override=herg_override,
            tox21_override=tox21_override,
        )

    def _provenance(self, provider: Any, policy: PredictionPolicySnapshot) -> dict[str, Any]:
        return {
            "request_id": str(uuid.uuid4()),
            "predictor_version": __version__,
            "policy_version": policy.policy_version,
            "tox21_task_order_version": TOX21_TASK_ORDER_VERSION,
            "models": [provider.model_id],
        }

    # -- prediction --------------------------------------------------------
    def predict(
        self,
        smiles: str,
        endpoints: Sequence[str] | None = None,
        *,
        herg_threshold_override: float | None = None,
        tox21_threshold_overrides: Mapping[str, float] | None = None,
    ) -> PredictionResult:
        results, errors = self.predict_batch(
            [smiles],
            endpoints,
            herg_threshold_override=herg_threshold_override,
            tox21_threshold_overrides=tox21_threshold_overrides,
        )
        if errors:
            raise InvalidSmilesError(smiles, errors[0].detail)
        return results[0]

    def predict_batch(
        self,
        smiles_list: Sequence[str],
        endpoints: Sequence[str] | None = None,
        *,
        herg_threshold_override: float | None = None,
        tox21_threshold_overrides: Mapping[str, float] | None = None,
    ) -> tuple[list[PredictionResult], list[BatchItemError]]:
        """Predict for a batch. Output order matches input order.

        Invalid items are reported per item; one bad SMILES does not fail the
        batch and does not shift the positions of the others.
        """
        if len(smiles_list) > self._max_batch_size:
            raise ValueError(
                f"batch of {len(smiles_list)} exceeds the limit of {self._max_batch_size}"
            )

        requested = self._resolve_endpoints(endpoints)
        provider = self._registry.for_capability(Endpoint.HERG.value)
        for endpoint in requested:
            other = self._registry.for_capability(endpoint.value)
            if other.model_id != provider.model_id:
                raise ArtifactError(
                    "this predictor build serves all requested endpoints from one provider; "
                    f"{endpoint.value} resolves to {other.model_id}"
                )

        molecules: list[Molecule | None] = []
        errors: list[BatchItemError] = []
        for index, raw in enumerate(smiles_list):
            try:
                molecules.append(resolve(raw))
            except InvalidSmilesError as exc:
                molecules.append(None)
                errors.append(
                    BatchItemError(
                        index=index,
                        input_smiles=raw if isinstance(raw, str) else "",
                        error="invalid_smiles",
                        detail=exc.reason,
                    )
                )

        valid = [(i, m) for i, m in enumerate(molecules) if m is not None]
        policy = self._policy(provider, herg_threshold_override, tox21_threshold_overrides)
        provenance = self._provenance(provider, policy)

        raw_rows: dict[int, dict[str, Any]] = {}
        if valid:
            outputs = provider.predict([m.canonical_smiles for _, m in valid])
            for (index, _), row in zip(valid, outputs):
                raw_rows[index] = row

        results: list[PredictionResult] = []
        for index, molecule in enumerate(molecules):
            if molecule is None:
                continue
            row = raw_rows[index]
            herg = tox21 = None
            if Endpoint.HERG in requested:
                herg = HergPrediction(
                    probability_blocker=row["herg_probability_blocker"],
                    threshold=policy.herg_threshold,
                    model_id=row["model_id"],
                )
            if Endpoint.TOX21 in requested:
                tox21 = Tox21Prediction(
                    assays=tuple(
                        Tox21AssayPrediction(
                            task=task,
                            probability_activity=row["tox21_probability_activity"][task],
                            threshold=policy.tox21_thresholds[task],
                        )
                        for task in TOX21_TASKS
                    ),
                    model_id=row["model_id"],
                )
            item_provenance = dict(provenance)
            item_provenance["truncated_input"] = bool(row.get("truncated", False))
            results.append(
                PredictionResult(
                    input_smiles=molecule.input_smiles,
                    canonical_smiles=molecule.canonical_smiles,
                    applicability=assess(molecule),
                    provenance=item_provenance,
                    herg=herg,
                    tox21=tox21,
                )
            )
        return results, errors

    # -- helpers -----------------------------------------------------------
    def _resolve_endpoints(self, endpoints: Iterable[str] | None) -> frozenset[Endpoint]:
        available = set(self._registry.describe_capabilities())
        if endpoints is None:
            resolved = {Endpoint(e) for e in sorted(available) if e in Endpoint._value2member_map_}
            if not resolved:
                raise ArtifactError("no endpoint is served by the current registry")
            return frozenset(resolved)

        resolved = set()
        for name in endpoints:
            try:
                endpoint = Endpoint(name)
            except ValueError:
                raise ValueError(
                    f"unknown endpoint {name!r}; known: {[e.value for e in Endpoint]}"
                ) from None
            if endpoint.value not in available:
                raise ArtifactError(
                    f"endpoint {endpoint.value!r} is not served by this build "
                    f"(available: {sorted(available)})"
                )
            resolved.add(endpoint)
        if not resolved:
            raise ValueError("at least one endpoint must be requested")
        return frozenset(resolved)
