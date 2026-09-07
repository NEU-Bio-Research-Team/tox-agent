"""Quick Predict — a stateless predictor call (plan section 3).

``frontend -> control-plane thin proxy -> ToxPred`` with no session, no run, no
analysis row, no observation, no outbox event, no agent runtime. The response is
the same ``AnalysisProjection`` shape the session path returns, built from an
in-memory snapshot that is never handed to the database.

Everything the session path does to keep the numbers honest still happens here:
endpoints are resolved through the same policy helper, threshold overrides go
through the same expert-role gate, the predictor's typed errors propagate
unchanged, and the provenance block is copied verbatim (SCI-10).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from ..config import PolicySettings
from ..domain.analysis import snapshot_from_prediction
from ..domain.ids import new_id
from ..predictor.client import PredictorClient
from . import projections
from .policy import Actor, authorise_threshold_overrides, policy_snapshot, resolve_endpoints


def _now() -> datetime:
    return datetime.now(timezone.utc)


class QuickPredict:
    def __init__(self, predictor: PredictorClient, policy: PolicySettings) -> None:
        self._predictor = predictor
        self._policy = policy

    async def execute(
        self,
        *,
        actor: Actor,
        smiles: str,
        endpoints: tuple[str, ...] | None = None,
        threshold_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return an ``AnalysisProjection``-shaped mapping. Never persisted:
        ``persisted`` is ``false`` and ``analysis_id`` is ``null`` so no caller
        mistakes this for something in their session history.
        """
        resolved = resolve_endpoints(endpoints, self._policy)
        overrides = authorise_threshold_overrides(threshold_overrides, actor, self._policy)
        policy = policy_snapshot(endpoints=resolved, overrides=overrides, actor=actor)

        response = await self._predictor.predict(
            smiles, resolved, threshold_overrides=overrides
        )
        provenance = self._predictor.provenance_of(response)

        # The snapshot needs identifiers to exist as a value object; these are
        # minted only to satisfy that constructor and are never written anywhere
        # or exposed — ``display_projection``'s ``analysis_id`` is overwritten
        # with ``null`` below, and it exposes no run or session id at all.
        snapshot = snapshot_from_prediction(
            session_id=new_id("ses"),
            run_id=new_id("run"),
            input_smiles=smiles,
            requested_endpoints=resolved,
            predictor_response=response.raw,
            provenance=provenance,
            policy_snapshot=policy,
            now=_now(),
        )

        return self._project(snapshot)

    async def execute_batch(
        self,
        *,
        actor: Actor,
        smiles: list[str],
        endpoints: tuple[str, ...] | None = None,
        threshold_overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Order-preserving batch. One unparseable molecule in the list comes
        back as a typed error at its own index; the rest still predict. Nothing
        is persisted."""
        resolved = resolve_endpoints(endpoints, self._policy)
        overrides = authorise_threshold_overrides(threshold_overrides, actor, self._policy)
        policy = policy_snapshot(endpoints=resolved, overrides=overrides, actor=actor)

        batch = await self._predictor.predict_batch(
            smiles, resolved, threshold_overrides=overrides
        )

        results = []
        for item in batch.results:
            snapshot = snapshot_from_prediction(
                session_id=new_id("ses"),
                run_id=new_id("run"),
                input_smiles=item.input_smiles,
                requested_endpoints=resolved,
                predictor_response=item.raw,
                provenance=self._predictor.provenance_of(item),
                policy_snapshot=policy,
                now=_now(),
            )
            results.append(self._project(snapshot))

        errors = [
            {
                "index": error.index,
                "input_smiles": error.input_smiles,
                "error": error.error,
                "detail": error.detail,
            }
            for error in batch.errors
        ]
        return {"results": results, "errors": errors, "count": len(smiles)}

    @staticmethod
    def _project(snapshot) -> dict[str, Any]:
        projection = projections.display_projection(snapshot)
        projection["persisted"] = False
        projection["analysis_id"] = None
        return projection
