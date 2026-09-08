"""AnalysisSnapshot — an immutable record of what the predictor said, once.

Plan section 5.4. The predictor response is stored losslessly after schema
validation and never rewritten; the UI and the model read projections of it.
The snapshot also pins what produced it — predictor version, git commit,
artifact hashes, and the resolved policy — so an answer written six months
later can still be explained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .ids import ANALYSIS, RUN, SESSION, new_id, require_id
from .provenance import content_sha256, idempotency_key


@dataclass(frozen=True, slots=True)
class PredictorProvenance:
    """Copied out of the predictor response verbatim (SCI-10). No model or
    projection code may rewrite these fields."""

    base_url_id: str
    service_version: str | None = None
    git_commit: str | None = None
    artifact_hashes: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictor_base_url_id": self.base_url_id,
            "predictor_service_version": self.service_version,
            "predictor_git_commit": self.git_commit,
            "artifact_hashes": list(self.artifact_hashes),
            "raw": dict(self.raw),
        }


@dataclass(frozen=True, slots=True)
class AnalysisSnapshot:
    id: str
    session_id: str
    run_id: str
    input_smiles: str
    canonical_smiles: str
    requested_endpoints: tuple[str, ...]
    predictor_response: dict[str, Any]
    provenance: PredictorProvenance
    policy_snapshot: dict[str, Any]
    content_sha256: str
    idempotency_key: str
    created_at: datetime

    def __post_init__(self) -> None:
        require_id(self.id, ANALYSIS, field="analysis.id")
        require_id(self.session_id, SESSION, field="analysis.session_id")
        require_id(self.run_id, RUN, field="analysis.run_id")
        if not self.canonical_smiles:
            raise ValueError("analysis.canonical_smiles is required")

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        run_id: str,
        input_smiles: str,
        requested_endpoints: tuple[str, ...],
        predictor_response: dict[str, Any],
        provenance: PredictorProvenance,
        policy_snapshot: dict[str, Any],
        now: datetime,
    ) -> "AnalysisSnapshot":
        canonical_smiles = predictor_response["canonical_smiles"]
        return cls(
            id=new_id(ANALYSIS),
            session_id=session_id,
            run_id=run_id,
            input_smiles=input_smiles,
            canonical_smiles=canonical_smiles,
            requested_endpoints=tuple(requested_endpoints),
            predictor_response=predictor_response,
            provenance=provenance,
            policy_snapshot=policy_snapshot,
            content_sha256=content_sha256(predictor_response),
            idempotency_key=snapshot_idempotency_key(
                canonical_smiles=canonical_smiles,
                endpoints=requested_endpoints,
                policy_snapshot=policy_snapshot,
                artifact_hashes=provenance.artifact_hashes,
            ),
            created_at=now,
        )

    @property
    def served_endpoints(self) -> tuple[str, ...]:
        return tuple(sorted(self.predictor_response.get("predictions", {})))

    @property
    def unavailable_endpoints(self) -> tuple[str, ...]:
        """Requested but absent. SCI-06 forbids filling these from elsewhere."""
        return tuple(sorted(set(self.requested_endpoints) - set(self.served_endpoints)))


def snapshot_from_prediction(
    *,
    session_id: str,
    run_id: str,
    input_smiles: str,
    requested_endpoints: tuple[str, ...],
    predictor_response: dict[str, Any],
    provenance: PredictorProvenance,
    policy_snapshot: dict[str, Any],
    now: datetime,
) -> "AnalysisSnapshot":
    """Build the immutable snapshot from a validated predictor response.

    Factored out of ``CreateAnalysis.execute`` so a caller that never persists —
    the stateless Quick Predict path — assembles the exact same snapshot object
    the display projection consumes, with no database round trip and no
    behaviour drift between the two paths. A pure construction: no I/O, no
    idempotency lookup, no run bookkeeping.
    """
    return AnalysisSnapshot.create(
        session_id=session_id,
        run_id=run_id,
        input_smiles=input_smiles,
        requested_endpoints=tuple(requested_endpoints),
        predictor_response=predictor_response,
        provenance=provenance,
        policy_snapshot=policy_snapshot,
        now=now,
    )


def snapshot_idempotency_key(
    *,
    canonical_smiles: str,
    endpoints: tuple[str, ...],
    policy_snapshot: dict[str, Any],
    artifact_hashes: tuple[str, ...],
) -> str:
    """Plan section 8.4: canonical SMILES + endpoints + resolved policy +
    predictor/artifact hash. A different threshold or a redeployed checkpoint is
    a different analysis, so neither may collapse onto an existing snapshot."""
    return idempotency_key(
        canonical_smiles, sorted(endpoints), policy_snapshot, sorted(artifact_hashes)
    )
