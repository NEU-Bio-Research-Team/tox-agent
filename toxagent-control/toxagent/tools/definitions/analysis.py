"""Analysis tools: snapshot, slice, attribution.

Each of these is the *only* way a model can obtain the corresponding facts, and
each returns field paths alongside values so that a claim written from the
result is a claim the validator can resolve. None of them accepts an owner or a
subject from the model: the session and run come from the capability token, and
an argument that disagrees with the token is a denial, not a preference (plan
section 8.5).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...application import projections
from ...application.create_analysis import CreateAnalysis
from ...domain.analysis import AnalysisSnapshot
from ...domain.errors import AnalysisNotFound, InvalidRequest, ToolDenied
from ...domain.events import EventType
from ...domain.observation import Observation, ObservationKind, Producer
from ...domain.provenance import idempotency_key
from ...predictor.client import PredictorClient
from ...predictor.contract import TOX21_TASKS
from ..registry import ToolContext, ToolDefinition, ToolOutput

Endpoint = Literal["clintox", "herg", "tox21"]


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSnapshotInput(_Input):
    session_id: str = Field(description="Must match the session this run belongs to.")
    smiles: str = Field(min_length=1, max_length=4000)
    endpoints: list[Endpoint] | None = Field(
        default=None,
        description="Endpoints to evaluate. An endpoint this deployment does not serve fails; "
                    "nothing is substituted for it.",
    )
    threshold_overrides: dict[str, Any] | None = Field(
        default=None,
        description="Only permitted for callers holding the expert role; usually null.",
    )


class SliceInput(_Input):
    analysis_id: str
    section: Literal["clintox", "herg", "tox21", "applicability", "provenance"]
    fields: list[str] | None = Field(
        default=None, description="Declared field names for this section. Omit for all of them."
    )
    task: str | None = Field(
        default=None,
        description="Required to read one Tox21 assay, e.g. 'SR-p53'. The twelve assays are "
                    "independent measurements.",
    )


class AttributionInput(_Input):
    analysis_id: str
    endpoint: Endpoint
    task: str | None = Field(
        default=None, description="Required when endpoint is 'tox21'."
    )


def _require_own_session(context: ToolContext, claimed: str) -> None:
    if claimed != context.session_id:
        raise ToolDenied(
            "the session in these arguments is not the session this run belongs to",
            run_id=context.run_id,
        )


async def _load(database, context: ToolContext, analysis_id: str) -> AnalysisSnapshot:
    async with database.unit_of_work() as uow:
        snapshot = await uow.analyses.get(analysis_id, session_id=context.session_id)
    if snapshot is None:
        # Scoped to the session, so a foreign analysis id is indistinguishable
        # from a nonexistent one.
        raise AnalysisNotFound("no such analysis in this session", analysis_id=analysis_id)
    return snapshot


def build(
    database,
    predictor: PredictorClient,
    create_analysis: CreateAnalysis,
) -> list[ToolDefinition]:
    async def create_snapshot(context: ToolContext, payload: CreateSnapshotInput) -> ToolOutput:
        _require_own_session(context, payload.session_id)
        result = await create_analysis.execute(
            actor=context.actor,
            session_id=context.session_id,
            run_id=context.run_id,
            smiles=payload.smiles,
            endpoints=tuple(payload.endpoints) if payload.endpoints else None,
            threshold_overrides=payload.threshold_overrides,
            # The run belongs to the turn that called this tool, not to the
            # snapshot; the answer still has to be written and validated.
            owns_run=False,
        )
        return ToolOutput(
            canonical=result.snapshot.predictor_response,
            model_view=result.model_view,
            ui_view=result.display,
            observation_ids=(result.observation.id,),
            provenance=result.snapshot.provenance.to_dict(),
        )

    async def analysis_slice(context: ToolContext, payload: SliceInput) -> ToolOutput:
        snapshot = await _load(database, context, payload.analysis_id)
        view = projections.slice_analysis(
            snapshot, payload.section, payload.fields, task=payload.task
        )
        async with database.unit_of_work() as uow:
            observations = await uow.observations.list_for_analysis(snapshot.id)
        source = next(
            (o for o in observations if o.kind is ObservationKind.PREDICTION), None
        )
        if source is None:
            raise AnalysisNotFound(
                "this analysis has no stored observation to cite", analysis_id=snapshot.id
            )
        # The slice cites the prediction observation rather than minting a new
        # one: the values are the same bytes, and a second observation of the
        # same fact would let two claims disagree about the same number.
        view["observation_id"] = source.id
        for entry in view["values"].values():
            entry["observation_id"] = source.id
        return ToolOutput(
            canonical=view, model_view=view, ui_view=view, observation_ids=(source.id,),
            provenance={"analysis_id": snapshot.id, "content_sha256": snapshot.content_sha256},
        )

    async def attribution(context: ToolContext, payload: AttributionInput) -> ToolOutput:
        snapshot = await _load(database, context, payload.analysis_id)
        if payload.endpoint == "tox21" and not payload.task:
            raise InvalidRequest(
                "attributing tox21 requires a task: the twelve assays are independent and a "
                "combined attribution would not mean anything",
                allowed=list(TOX21_TASKS),
            )
        if payload.endpoint != "tox21" and payload.task:
            raise InvalidRequest(f"task is only meaningful for tox21, not {payload.endpoint}")
        if payload.endpoint not in snapshot.served_endpoints:
            raise InvalidRequest(
                f"this analysis has no {payload.endpoint} section",
                served=list(snapshot.served_endpoints),
            )

        cache_key = idempotency_key(
            "attribution", snapshot.canonical_smiles, payload.endpoint, payload.task,
            sorted(snapshot.provenance.artifact_hashes),
        )
        async with database.unit_of_work() as uow:
            for existing in await uow.observations.list_for_analysis(snapshot.id):
                if (
                    existing.kind is ObservationKind.ATTRIBUTION
                    and existing.provenance.get("cache_key") == cache_key
                ):
                    return ToolOutput(
                        canonical=existing.canonical_payload,
                        model_view=existing.model_projection,
                        ui_view=existing.canonical_payload,
                        observation_ids=(existing.id,),
                        provenance={**existing.provenance, "cached": True},
                    )

        response = await predictor.attribution(
            snapshot.canonical_smiles, payload.endpoint, payload.task
        )
        if response.status == "failed":
            raise InvalidRequest(
                f"attribution failed: {response.message or response.error}",
                endpoint=payload.endpoint, task=payload.task,
            )

        canonical = response.model_dump(mode="json")
        model_view = {
            "analysis_id": snapshot.id,
            "endpoint": payload.endpoint,
            "task": payload.task,
            "status": response.status,
            "method": response.metadata.get("method"),
            "model_id": response.metadata.get("model_id"),
            # Top contributions only. The full token list is unbounded and the
            # tail is noise a model would spend budget reading.
            "top_tokens": sorted(
                response.tokens, key=lambda t: abs(float(t.get("score", 0.0))), reverse=True
            )[:12],
            "required_limitations": ["attribution_not_causality"],
        }
        observation = Observation.create(
            session_id=context.session_id,
            run_id=context.run_id,
            producer=Producer.ATTRIBUTION,
            kind=ObservationKind.ATTRIBUTION,
            schema_version="toxpred-attribution-v1",
            canonical_payload=canonical,
            model_projection=model_view,
            provenance={
                "analysis_id": snapshot.id,
                "cache_key": cache_key,
                "method": response.metadata.get("method"),
                "model_id": response.metadata.get("model_id"),
                **snapshot.provenance.to_dict(),
            },
            now=_now(),
            required_limitations=("attribution_not_causality",),
        )
        async with database.unit_of_work() as uow:
            await uow.observations.add(observation, analysis_id=snapshot.id)
            uow.emit(
                session_id=context.session_id, type=EventType.OBSERVATION_CREATED,
                entity_type="observation", entity_id=observation.id, run_id=context.run_id,
                payload={"kind": "attribution", "endpoint": payload.endpoint},
            )
            await uow.commit()

        return ToolOutput(
            canonical=canonical, model_view=observation.model_projection, ui_view=canonical,
            observation_ids=(observation.id,), provenance=observation.provenance,
        )

    return [
        ToolDefinition(
            name="create_analysis_snapshot",
            title="Create an analysis snapshot",
            description=(
                "Run the toxicity predictor on one SMILES and store the result as an immutable "
                "snapshot. Returns which sections exist and which limitations any answer using "
                "them must carry. Does not perform attribution or literature search."
            ),
            input_model=CreateSnapshotInput,
            handler=create_snapshot,
            profiles=frozenset({"analysis"}),
            soft_timeout_s=60.0,
            hard_timeout_s=120.0,
            max_retries=1,
        ),
        ToolDefinition(
            name="get_analysis_slice",
            title="Read declared fields of an analysis",
            description=(
                "Return specific fields of one section of a stored analysis, each with the field "
                "path and observation id needed to cite it. hERG, Tox21 and ClinTox are separate "
                "measurements and are never interchangeable."
            ),
            input_model=SliceInput,
            handler=analysis_slice,
            profiles=frozenset({"analysis", "report_qa", "evidence_research", "audit_readonly"}),
            soft_timeout_s=2.0,
            hard_timeout_s=5.0,
        ),
        ToolDefinition(
            name="get_attribution",
            title="Explain one endpoint's score",
            description=(
                "Return per-token importance for exactly one endpoint, and for Tox21 exactly one "
                "assay. Attribution shows what moved the model's score; it is not evidence of a "
                "chemical mechanism."
            ),
            input_model=AttributionInput,
            handler=attribution,
            profiles=frozenset({"report_qa"}),
            soft_timeout_s=90.0,
            hard_timeout_s=180.0,
        ),
    ]
