"""Create an analysis snapshot (plan section 7.1).

Deterministic end to end. No model is consulted, no research is performed, and
no attribution is computed unless it was asked for separately — an analysis that
quietly ran three extra things is an analysis whose cost and latency nobody can
predict.

The predictor call happens outside the database transaction. Holding a
transaction open across a two-minute forward pass would block every other write
in the session for the duration; the write that follows is short and atomic.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from ...platform.config import PolicySettings
from ...domain.analysis import (
    AnalysisSnapshot,
    snapshot_from_prediction,
    snapshot_idempotency_key,
)
from ...domain.errors import SessionNotFound
from ...domain.events import EventType
from ...domain.observation import Observation, ObservationKind, Producer
from ...domain.run import RunStatus
from ...domain.session import Session
from ...predictor.client import PredictorClient
from ..conversation import projections
from ..explanation.identity import (
    explanation_checkpoint_key,
    model_artifact_fingerprint,
)
from ..policy import Actor, authorise_threshold_overrides, policy_snapshot, resolve_endpoints
from ..runs.transitions import advance


log = logging.getLogger("toxagent.analysis")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _failed_explanation(
    *, endpoint: str, task: str | None, smiles: str, canonical_smiles: str, reason: str
) -> dict[str, Any]:
    """A terminal artifact saying an explanation was not produced.

    Never approximate data, never a silent omission: a missing target reads as
    "not requested", and an unlabelled one as "here is the attribution".
    """
    return {
        "status": "failed", "endpoint": endpoint, "task": task,
        "input_smiles": smiles, "canonical_smiles": canonical_smiles,
        "atoms": [], "bonds": [], "tokens": [], "method": "unavailable",
        "metadata": {"error": reason},
    }


@dataclass(frozen=True, slots=True)
class ExplanationOutcome:
    """One computed target, carrying the identity it was computed under.

    The cache key travels with the payload because it is what makes the
    explanation findable later. Before XAI-01 this loop returned bare payloads,
    the key stayed inside the checkpoint table, and the observation the report
    builder reads had no key on it at all — so the report could not tell that
    the attribution in front of it was the one the analysis had already paid
    for, and asked the predictor again.
    """

    payload: dict[str, Any]
    cache_key: str
    endpoint: str
    task: str | None
    model_id: str | None


@dataclass(frozen=True)
class AnalysisResult:
    snapshot: AnalysisSnapshot
    observation: Observation
    reused: bool

    @property
    def display(self) -> dict[str, Any]:
        return projections.display_projection(self.snapshot)

    @property
    def model_view(self) -> dict[str, Any]:
        return projections.model_projection(self.snapshot)


class CreateAnalysis:
    def __init__(self, database, predictor: PredictorClient, settings: PolicySettings) -> None:
        self._db = database
        self._predictor = predictor
        self._settings = settings

    @staticmethod
    def _resolved_model(response, endpoint: str, requested: Mapping[str, str] | None) -> str | None:
        """The model that actually produced this endpoint's probability.

        Prefer the response over the request: `model_selection` says what was
        asked for, `predictions.<endpoint>.model_id` says what answered, and
        an explanation has to be about the latter. Falls back to the request
        only when the predictor reported none, and to None when neither knows
        — at which point the predictor auto-resolves, and refuses outright if
        the endpoint has more than one admitted model.
        """
        prediction = getattr(response.predictions, endpoint, None)
        resolved = getattr(prediction, "model_id", None) if prediction is not None else None
        if resolved:
            return resolved
        return (requested or {}).get(endpoint)

    async def execute(
        self,
        *,
        actor: Actor,
        session_id: str,
        run_id: str,
        smiles: str,
        endpoints: tuple[str, ...] | None = None,
        model_selection: Mapping[str, str] | None = None,
        threshold_overrides: Mapping[str, Any] | None = None,
        explanation_mode: str = "on_demand",
        explanation_targets: tuple[tuple[str, str | None], ...] = (),
        owns_run: bool = True,
    ) -> AnalysisResult:
        """``owns_run`` is False when this runs as a tool inside a larger run.

        The deterministic lane drives the run to completion here; an agentic run
        that snapshots a molecule mid-turn does not finish when the snapshot
        does, and completing it would strand the answer that was about to be
        written.
        """
        endpoints = resolve_endpoints(endpoints, self._settings)
        overrides = authorise_threshold_overrides(threshold_overrides, actor, self._settings)
        policy = policy_snapshot(
            endpoints=endpoints, overrides=overrides, actor=actor,
            explanation_mode=explanation_mode, explanation_targets=explanation_targets,
        )

        async with self._db.unit_of_work() as uow:
            session = await uow.sessions.get(session_id, owner_id=actor.subject_id)
            if session is None:
                raise SessionNotFound("no such session", session_id=session_id)
            run = await uow.runs.get(run_id)
            if owns_run and run is not None and run.status is RunStatus.QUEUED:
                await advance(uow, run, RunStatus.RUNNING)
                await uow.commit()

        response = await self._predictor.predict(
            smiles, endpoints, model_selection=model_selection, threshold_overrides=overrides
        )
        provenance = self._predictor.provenance_of(response)
        explanations: list[ExplanationOutcome] = []
        if explanation_mode == "required":
            explanations = await self._explain_targets(
                session_id=session_id,
                smiles=smiles,
                response=response,
                provenance=provenance,
                model_selection=model_selection,
                targets=explanation_targets,
            )

        # Idempotency is checked after the call rather than before it: the key
        # is defined over the *canonical* SMILES, which only the predictor can
        # produce. Two spellings of one molecule therefore cost two forward
        # passes but produce one snapshot, which is the direction of error that
        # keeps the audit trail honest.
        key = snapshot_idempotency_key(
            canonical_smiles=response.canonical_smiles,
            endpoints=endpoints,
            policy_snapshot=policy,
            artifact_hashes=provenance.artifact_hashes,
        )

        async with self._db.unit_of_work() as uow:
            existing = await uow.analyses.find_by_idempotency_key(session_id, key)
            if existing is not None:
                observations = await uow.observations.list_for_analysis(existing.id)
                await self._complete(uow, session, run_id, existing, reused=True, owns_run=owns_run)
                await uow.commit()
                return AnalysisResult(existing, observations[0], reused=True)

            snapshot = snapshot_from_prediction(
                session_id=session_id,
                run_id=run_id,
                input_smiles=smiles,
                requested_endpoints=endpoints,
                predictor_response=response.raw,
                provenance=provenance,
                policy_snapshot=policy,
                now=_now(),
            )
            observation = self._observation_for(snapshot, run_id)
            await uow.analyses.add(snapshot)
            await uow.observations.add(observation, analysis_id=snapshot.id)
            uow.emit(
                session_id=session_id, type=EventType.ANALYSIS_CREATED,
                entity_type="analysis", entity_id=snapshot.id, run_id=run_id,
                payload={"canonical_smiles": snapshot.canonical_smiles},
            )
            uow.emit(
                session_id=session_id, type=EventType.OBSERVATION_CREATED,
                entity_type="observation", entity_id=observation.id, run_id=run_id,
                payload={"kind": observation.kind.value},
            )
            for outcome in explanations:
                explanation_observation = self._explanation_observation(snapshot, run_id, outcome)
                await uow.observations.add(explanation_observation, analysis_id=snapshot.id)
                uow.emit(
                    session_id=session_id, type=EventType.OBSERVATION_CREATED,
                    entity_type="observation", entity_id=explanation_observation.id, run_id=run_id,
                    payload={
                        "kind": "attribution",
                        "endpoint": outcome.endpoint,
                        "task": outcome.task,
                        "status": outcome.payload.get("status"),
                    },
                )
            await self._complete(uow, session, run_id, snapshot, reused=False, owns_run=owns_run)
            await uow.commit()

        return AnalysisResult(snapshot, observation, reused=False)

    async def _explain_targets(
        self,
        *,
        session_id: str,
        smiles: str,
        response,
        provenance,
        model_selection: Mapping[str, str] | None,
        targets: tuple[tuple[str, str | None], ...],
    ) -> list[ExplanationOutcome]:
        """Compute each requested explanation, committing them as they land.

        I19: this used to be a loop that computed every target and then wrote
        the lot at the end, with no bound on how long the loop took. Two
        consequences, both invisible until something went wrong:

        - A crash after the fifth of eight targets discarded all five. The
          retry recomputed them, at the same cost, having learnt nothing.
        - Each request had the predictor client's own timeout, and nothing had
          a timeout for the sequence. Eight healthy-looking calls could
          together outlast the run deadline they were supposed to fit inside.

        Now each target is checkpointed the moment it succeeds, keyed by what
        it explains rather than by which run asked, so a retry — or another
        run asking the same question of the same molecule and model — reuses
        it. And the whole sequence runs under one budget: a target not reached
        inside it is recorded as failed with that reason, which is a true
        statement about what happened, unlike an absent target or an
        unlabelled approximation.
        """
        keys = {}
        for endpoint, task in targets:
            # I11: the same model that produced the probability, not whichever
            # one the predictor would auto-resolve. With one admitted model per
            # endpoint these agree; with two they did not, and the bundle
            # paired B's number with A's attribution while claiming both were
            # about B.
            model_id = self._resolved_model(response, endpoint, model_selection)
            keys[(endpoint, task)] = explanation_checkpoint_key(
                canonical_smiles=response.canonical_smiles,
                endpoint=endpoint, task=task, model_id=model_id,
                artifact_fingerprint=model_artifact_fingerprint(provenance, model_id),
            )
        async with self._db.unit_of_work() as uow:
            done = await uow.explanation_checkpoints.get_many(list(keys.values()))

        deadline = time.monotonic() + self._settings.explanation_budget_s
        explanations: list[ExplanationOutcome] = []
        for endpoint, task in targets:
            key = keys[(endpoint, task)]
            model_id = self._resolved_model(response, endpoint, model_selection)

            # Both closures are called within this iteration only, so binding
            # the loop variables late is correct here.
            def outcome(payload: dict[str, Any]) -> ExplanationOutcome:  # noqa: B023
                return ExplanationOutcome(
                    payload=payload, cache_key=key,  # noqa: B023
                    endpoint=endpoint, task=task, model_id=model_id,  # noqa: B023
                )

            def failed(reason: str) -> ExplanationOutcome:
                return outcome(_failed_explanation(
                    endpoint=endpoint, task=task, smiles=smiles,  # noqa: B023
                    canonical_smiles=response.canonical_smiles, reason=reason,
                ))

            committed = done.get(key)
            if committed is not None:
                # Already paid for, by an earlier attempt at this run or by
                # another run asking the same question.
                explanations.append(outcome(committed))
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                explanations.append(failed("explanation_budget_exceeded"))
                continue
            try:
                explanation = await asyncio.wait_for(
                    self._predictor.explain(
                        response.canonical_smiles, endpoint, task, model_id=model_id,
                    ),
                    timeout=remaining,
                )
                payload = explanation.model_dump(mode="json")
            except asyncio.TimeoutError:
                explanations.append(failed("explanation_budget_exceeded"))
                continue
            except asyncio.CancelledError:
                # The run itself is being cancelled. Whatever has been
                # checkpointed stays checkpointed; nothing here should turn a
                # cancellation into a fabricated failure artifact.
                raise
            except Exception as exc:  # terminal failure artifact; never fabricate XAI
                explanations.append(failed(type(exc).__name__))
                continue

            explanations.append(outcome(payload))
            # Committed on its own, immediately: a checkpoint written in the
            # same transaction as the snapshot would be lost by exactly the
            # crash it exists to survive. A failure to record it costs a
            # recomputation later, which is why it does not fail the analysis.
            try:
                async with self._db.unit_of_work() as uow:
                    await uow.explanation_checkpoints.put(
                        key, session_id=session_id, endpoint=endpoint, task=task,
                        model_id=model_id, canonical_smiles=response.canonical_smiles,
                        payload=payload, now=_now(),
                    )
                    await uow.commit()
            except Exception:  # noqa: BLE001
                log.exception(
                    "could not checkpoint the %s explanation for session %s", endpoint, session_id
                )
        return explanations

    # --- helpers -----------------------------------------------------------

    @staticmethod
    def _observation_for(snapshot: AnalysisSnapshot, run_id: str) -> Observation:
        """One observation per snapshot, holding the response losslessly.

        The canonical payload is the whole predictor response, so a claim's
        field path is a path into what the predictor actually said — not into a
        reshaped copy whose field names this layer chose.
        """
        return Observation.create(
            session_id=snapshot.session_id,
            run_id=run_id,
            producer=Producer.PREDICTOR,
            kind=ObservationKind.PREDICTION,
            schema_version="toxpred-prediction-v1",
            canonical_payload=snapshot.predictor_response,
            model_projection=projections.model_projection(snapshot),
            provenance={
                **snapshot.provenance.to_dict(),
                "analysis_id": snapshot.id,
                "content_sha256": snapshot.content_sha256,
            },
            now=snapshot.created_at,
            required_limitations=projections.required_limitations(snapshot),
        )

    @staticmethod
    def _explanation_observation(
        snapshot: AnalysisSnapshot, run_id: str, outcome: ExplanationOutcome
    ) -> Observation:
        """The observation both pipelines write, under one schema version.

        The projection and the provenance are built by ``application.explanation``
        rather than here, so an explanation produced eagerly by an analysis and
        one produced on demand by ``get_or_create_explanation`` are the same row
        shape. That is what lets the report builder resolve either of them, and
        what stops it recomputing an attribution it is already holding.

        No figure is drawn here. This path has neither an object store nor an
        owner id, and the honest consequence is a numeric explanation with no
        picture yet; ``GetOrCreateExplanation`` draws it later from *this*
        payload, without a second backward pass.
        """
        from ..explanation.service import explanation_observation

        return explanation_observation(
            payload=outcome.payload,
            session_id=snapshot.session_id,
            run_id=run_id,
            analysis_id=snapshot.id,
            snapshot_provenance=snapshot.provenance.to_dict(),
            cache_key=outcome.cache_key,
            requested_model_id=outcome.model_id,
            endpoint=outcome.endpoint,
            task=outcome.task,
            now=_now(),
        )

    @staticmethod
    async def _complete(
        uow,
        session: Session,
        run_id: str,
        snapshot: AnalysisSnapshot,
        *,
        reused: bool,
        owns_run: bool = True,
    ) -> None:
        run = await uow.runs.get(run_id)
        if owns_run and run is not None and not run.is_terminal:
            await advance(
                uow, run, RunStatus.COMPLETED,
                payload={"analysis_id": snapshot.id, "reused_snapshot": reused},
            )
        current = await uow.sessions.get_unscoped(session.id)
        if current is not None and current.active_analysis_id != snapshot.id:
            await uow.sessions.update(
                current.with_active_analysis(snapshot.id, now=_now()),
                expected_version=current.version,
            )


@dataclass(frozen=True)
class BatchResult:
    snapshots: tuple[AnalysisSnapshot, ...]
    failures: tuple[dict[str, Any], ...]

    @property
    def count(self) -> int:
        return len(self.snapshots) + len(self.failures)


class CreateAnalysisBatch:
    """Batch analysis (UC-02).

    Order is preserved and failures are per item: one unparseable molecule in a
    list of fifty does not fail the other forty-nine, and it does not become a
    prediction either — it comes back as a typed error at its own index.
    """

    def __init__(self, database, predictor: PredictorClient, settings: PolicySettings) -> None:
        self._db = database
        self._predictor = predictor
        self._settings = settings

    async def execute(
        self,
        *,
        actor: Actor,
        session_id: str,
        run_id: str,
        smiles: list[str],
        endpoints: tuple[str, ...] | None = None,
        model_selection: Mapping[str, str] | None = None,
        threshold_overrides: Mapping[str, Any] | None = None,
    ) -> BatchResult:
        endpoints = resolve_endpoints(endpoints, self._settings)
        overrides = authorise_threshold_overrides(threshold_overrides, actor, self._settings)
        policy = policy_snapshot(endpoints=endpoints, overrides=overrides, actor=actor)

        async with self._db.unit_of_work() as uow:
            session = await uow.sessions.get(session_id, owner_id=actor.subject_id)
            if session is None:
                raise SessionNotFound("no such session", session_id=session_id)

        response = await self._predictor.predict_batch(
            smiles, endpoints, model_selection=model_selection, threshold_overrides=overrides
        )

        stored: list[AnalysisSnapshot] = []
        async with self._db.unit_of_work() as uow:
            for item in response.results:
                provenance = self._predictor.provenance_of(item)
                key = snapshot_idempotency_key(
                    canonical_smiles=item.canonical_smiles, endpoints=endpoints,
                    policy_snapshot=policy, artifact_hashes=provenance.artifact_hashes,
                )
                existing = await uow.analyses.find_by_idempotency_key(session_id, key)
                if existing is not None:
                    stored.append(existing)
                    continue
                snapshot = snapshot_from_prediction(
                    session_id=session_id, run_id=run_id, input_smiles=item.input_smiles,
                    requested_endpoints=endpoints, predictor_response=item.raw,
                    provenance=provenance, policy_snapshot=policy, now=_now(),
                )
                await uow.analyses.add(snapshot)
                await uow.observations.add(
                    CreateAnalysis._observation_for(snapshot, run_id), analysis_id=snapshot.id
                )
                uow.emit(
                    session_id=session_id, type=EventType.ANALYSIS_CREATED,
                    entity_type="analysis", entity_id=snapshot.id, run_id=run_id,
                    payload={"canonical_smiles": snapshot.canonical_smiles},
                )
                stored.append(snapshot)

            run = await uow.runs.get(run_id)
            if run is not None and not run.is_terminal:
                await advance(
                    uow, run, RunStatus.COMPLETED,
                    payload={
                        "analysis_ids": [s.id for s in stored],
                        "failed": len(response.errors),
                    },
                )
            await uow.commit()

        return BatchResult(
            tuple(stored),
            tuple(
                {
                    "index": e.index, "input_smiles": e.input_smiles,
                    "error": e.error, "detail": e.detail,
                }
                for e in response.errors
            ),
        )
