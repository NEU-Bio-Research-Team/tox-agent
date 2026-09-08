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
    ClinToxPrediction,
    HergPrediction,
    PredictionResult,
    Tox21AssayPrediction,
    Tox21Prediction,
)
from ..scientific.applicability import assess
from ..scientific.artifacts import ArtifactError
from ..scientific.featurization.rdkit_resolver import resolve
from ..scientific.registry import ModelRegistry
from ..scientific.providers.contracts import (
    ClinToxRawOutput,
    HergTox21RawOutput,
    ProviderBatchResult,
)

MAX_BATCH_SIZE = 256


@dataclass(frozen=True)
class BatchItemError:
    index: int
    input_smiles: str
    error: str
    detail: str


class ToxicityPredictor:
    def __init__(self, registry: ModelRegistry, *, max_batch_size: int = MAX_BATCH_SIZE,
                 calibrators: Mapping[str, Any] | None = None) -> None:
        self._registry = registry
        self._max_batch_size = int(max_batch_size)
        self._calibrators = dict(calibrators or {})

    def _calibrated(self, endpoint: str, probability: float, *, task: str | None = None) -> float | None:
        calibrator = self._calibrators.get(endpoint)
        return None if calibrator is None else float(calibrator.calibrate(probability, task=task))

    # -- policy ------------------------------------------------------------
    def _policy(
        self,
        providers: Mapping[Endpoint, Any],
        herg_override: float | None,
        tox21_override: Mapping[str, float] | None,
        clintox_override: float | None,
    ) -> PredictionPolicySnapshot:
        dual = providers.get(Endpoint.HERG) or providers.get(Endpoint.TOX21)
        if dual is None:
            # The v1 policy snapshot carries all endpoint operating points.
            # Resolve the dual-head policy owner without executing it so a
            # ClinTox-only request does not depend on flattened provider rows.
            try:
                dual = self._registry.for_capability(Endpoint.HERG.value)
            except ArtifactError:
                dual = None
        if dual is None:
            raise ArtifactError(
                "this build resolves hERG and Tox21 thresholds from the dual-head artifact; "
                "neither endpoint is available"
            )

        clintox_threshold = None
        if Endpoint.CLINTOX in providers:
            declared = self._registry.spec(
                providers[Endpoint.CLINTOX].model_id
            ).declared_thresholds
            clintox_threshold = declared.get("clintox")
            if clintox_threshold is None:
                raise ArtifactError(
                    "clintox is served but no threshold is declared for it; refusing to "
                    "invent an operating point"
                )

        return PredictionPolicySnapshot.from_artifact(
            herg_threshold=dual.artifact_herg_threshold,
            tox21_thresholds=dual.artifact_tox21_thresholds,
            clintox_threshold=clintox_threshold,
            herg_override=herg_override,
            tox21_override=tox21_override,
            clintox_override=clintox_override,
        )

    def _provenance(
        self, providers: Mapping[Endpoint, Any], policy: PredictionPolicySnapshot
    ) -> dict[str, Any]:
        """Everything needed to reproduce this answer later.

        Plan section 4.4 requires model_id, artifact SHA-256, base-model
        revision, tokenizer revision and policy version to be present on every
        response — so a number in a report can be traced to the exact weights
        that produced it.
        """
        artifacts = []
        for model_id in sorted({p.model_id for p in providers.values()}):
            spec = self._registry.spec(model_id)
            weights = next(
                (f for f in spec.files if f.relative_path.endswith((".pt", ".safetensors"))),
                None,
            )
            tokenizer = next(
                (f for f in spec.files if f.relative_path.endswith("tokenizer.json")), None
            )
            artifacts.append(
                {
                    "model_id": model_id,
                    "weights_sha256": weights.sha256 if weights else None,
                    "tokenizer_sha256": tokenizer.sha256 if tokenizer else None,
                    "feature_schema_version": spec.feature_schema_version,
                    "base_model": dict(spec.base_model) or None,
                }
            )
        return {
            "request_id": str(uuid.uuid4()),
            "predictor_version": __version__,
            "policy_version": policy.policy_version,
            "tox21_task_order_version": TOX21_TASK_ORDER_VERSION,
            "models": sorted({p.model_id for p in providers.values()}),
            "artifacts": artifacts,
        }

    # -- prediction --------------------------------------------------------
    def predict(
        self,
        smiles: str,
        endpoints: Sequence[str] | None = None,
        *,
        herg_threshold_override: float | None = None,
        tox21_threshold_overrides: Mapping[str, float] | None = None,
        clintox_threshold_override: float | None = None,
    ) -> PredictionResult:
        results, errors = self.predict_batch(
            [smiles],
            endpoints,
            herg_threshold_override=herg_threshold_override,
            tox21_threshold_overrides=tox21_threshold_overrides,
            clintox_threshold_override=clintox_threshold_override,
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
        clintox_threshold_override: float | None = None,
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
        providers = {e: self._registry.for_capability(e.value) for e in requested}

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
        valid_positions = {source_index: position for position, (source_index, _) in enumerate(valid)}
        policy = self._policy(
            providers,
            herg_threshold_override,
            tox21_threshold_overrides,
            clintox_threshold_override,
        )
        provenance = self._provenance(providers, policy)

        # One call per distinct provider, not per endpoint: the dual-head model
        # returns hERG and Tox21 together, so asking for both must not run the
        # backbone twice.
        provider_outputs: dict[str, Sequence[Any]] = {}
        if valid:
            canonical = [m.canonical_smiles for _, m in valid]
            for provider in {p.model_id: p for p in providers.values()}.values():
                outputs = provider.predict(canonical)
                if len(outputs) != len(valid):
                    raise ArtifactError(
                        f"[{provider.model_id}] returned {len(outputs)} rows for "
                        f"{len(valid)} molecules"
                    )
                if provider.model_id in provider_outputs:
                    raise ArtifactError(f"duplicate provider output for {provider.model_id!r}")
                provider_outputs[provider.model_id] = outputs

        results: list[PredictionResult] = []
        for index, molecule in enumerate(molecules):
            if molecule is None:
                continue
            valid_index = valid_positions[index]
            clintox = herg = tox21 = None
            if Endpoint.CLINTOX in requested:
                owner = providers[Endpoint.CLINTOX]
                row = provider_outputs[owner.model_id][valid_index]
                if not isinstance(row, ClinToxRawOutput):
                    raise ArtifactError(f"[{owner.model_id}] returned wrong typed output")
                clintox = ClinToxPrediction(
                    probability_clinical_toxicity=row.clintox_probability_toxicity,
                    threshold=policy.clintox_threshold,
                    model_id=owner.model_id,
                    calibrated_probability_clinical_toxicity=self._calibrated(
                        "clintox", row.clintox_probability_toxicity
                    ),
                )
            if Endpoint.HERG in requested:
                owner = providers[Endpoint.HERG]
                row = provider_outputs[owner.model_id][valid_index]
                if not isinstance(row, HergTox21RawOutput):
                    raise ArtifactError(f"[{owner.model_id}] returned wrong typed output")
                herg = HergPrediction(
                    probability_blocker=row.herg_probability_blocker,
                    threshold=policy.herg_threshold,
                    model_id=owner.model_id,
                    calibrated_probability_blocker=self._calibrated(
                        "herg", row.herg_probability_blocker
                    ),
                )
            if Endpoint.TOX21 in requested:
                owner = providers[Endpoint.TOX21]
                row = provider_outputs[owner.model_id][valid_index]
                if not isinstance(row, HergTox21RawOutput):
                    raise ArtifactError(f"[{owner.model_id}] returned wrong typed output")
                tox21 = Tox21Prediction(
                    assays=tuple(
                        Tox21AssayPrediction(
                            task=task,
                            probability_activity=row.tox21_probability_activity[task],
                            threshold=policy.tox21_thresholds[task],
                            calibrated_probability_activity=self._calibrated(
                                "tox21", row.tox21_probability_activity[task], task=task
                            ),
                        )
                        for task in TOX21_TASKS
                    ),
                    model_id=owner.model_id,
                )
            item_provenance = dict(provenance)
            item_provenance["schema_version"] = "toxpred-provenance-v2"
            token_owner = providers.get(Endpoint.HERG) or providers.get(Endpoint.TOX21)
            if token_owner is not None:
                token_row = provider_outputs[token_owner.model_id][valid_index]
                if isinstance(token_row, HergTox21RawOutput):
                    item_provenance["tokenization"] = token_row.tokenization.to_dict()
                    item_provenance["truncated_input"] = token_row.tokenization.truncated
            results.append(
                PredictionResult(
                    input_smiles=molecule.input_smiles,
                    canonical_smiles=molecule.canonical_smiles,
                    applicability=assess(molecule),
                    provenance=item_provenance,
                    clintox=clintox,
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
