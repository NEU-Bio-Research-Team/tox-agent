"""Explanation-package tools (report spec section 8).

Two tools, one deliberate split:

``get_or_create_explanation`` does the expensive, billable thing — a predictor
backward pass — and is therefore the only one that can create anything. It
returns *refs*: an explanation id, a figure id and the ranked contributors. It
never returns SVG bytes, never returns an attachment URL, and never exposes the
renderer, because a runtime that could reach figure storage directly could put
an image into a report that no explanation observation accounts for
(spec section 8: "Do not expose both low-level figure rendering and attachment
storage to the agent").

``get_explanation_package`` re-reads one that already exists. It calls no
provider and creates nothing, so a model re-checking a contributor list during
drafting cannot spend the run's predictor budget on it.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ...application.explanation import GetOrCreateExplanation, package_from_observation
from ...domain.errors import AnalysisNotFound, InvalidRequest
from ...domain.observation import ObservationKind
from ..registry import ToolContext, ToolDefinition, ToolOutput

Endpoint = Literal["clintox", "herg", "tox21"]


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetOrCreateExplanationInput(_Input):
    analysis_id: str
    endpoint: Endpoint
    task: str | None = Field(
        default=None,
        description=(
            "Required when endpoint is 'tox21', e.g. 'SR-p53'. The twelve assays are "
            "independent measurements and one combined explanation of them would not "
            "mean anything."
        ),
    )
    top_k: int = Field(
        default=8, ge=1, le=25,
        description="How many contributors to return per direction, ranked by magnitude.",
    )


class GetExplanationPackageInput(_Input):
    #: Always required, even when explanation_id identifies the package on its
    #: own: it is what scopes the lookup to an analysis this session owns, the
    #: same way every other read tool is scoped. An id alone would make a
    #: foreign explanation distinguishable from a nonexistent one.
    analysis_id: str
    explanation_id: str | None = Field(
        default=None,
        description="The id returned by get_or_create_explanation. Give this, or an "
                    "endpoint (and task for Tox21).",
    )
    endpoint: Endpoint | None = None
    task: str | None = None
    top_k: int = Field(default=8, ge=1, le=25)


def _package_view(package, *, analysis_id: str) -> dict:
    """The bounded projection. Contributor entries carry atom indices and
    signed contributions — enough to write an honest narrative — and no bytes."""
    figure = package.figure
    return {
        "analysis_id": analysis_id,
        "explanation_id": package.explanation_id,
        "observation_id": package.observation_id,
        "endpoint": package.endpoint,
        "task": package.task,
        "method": package.method,
        "status": package.status.value,
        "figure": (
            {
                "figure_id": figure.figure_id,
                "media_type": figure.media_type,
                "caption": figure.caption,
                "alt_text": figure.alt_text,
                "content_sha256": figure.content_sha256,
            }
            if figure
            else None
        ),
        "figure_unavailable_reason": None if figure else package.failure_reason,
        "extracted_highlights": package.highlights.to_dict(),
        "required_limitations": list(package.required_limitations),
    }


def build(database, predictor, object_store=None) -> list[ToolDefinition]:
    service = GetOrCreateExplanation(database, predictor, object_store)

    async def get_or_create(
        context: ToolContext, payload: GetOrCreateExplanationInput
    ) -> ToolOutput:
        result = await service.execute(
            owner_id=context.actor.subject_id,
            session_id=context.session_id,
            run_id=context.run_id,
            analysis_id=payload.analysis_id,
            endpoint=payload.endpoint,
            task=payload.task,
            top_k=payload.top_k,
        )
        view = _package_view(result.package, analysis_id=payload.analysis_id)
        view["reused"] = result.reused
        return ToolOutput(
            canonical=result.observation.canonical_payload,
            model_view=view,
            ui_view=result.package.to_dict(),
            observation_ids=(result.observation.id,),
            provenance=result.observation.provenance,
        )

    async def get_package(
        context: ToolContext, payload: GetExplanationPackageInput
    ) -> ToolOutput:
        if not payload.explanation_id and not payload.endpoint:
            raise InvalidRequest(
                "name an explanation_id, or an endpoint (and a task for Tox21)"
            )
        async with database.unit_of_work() as uow:
            snapshot = await uow.analyses.get(
                payload.analysis_id, session_id=context.session_id
            )
            if snapshot is None:
                raise AnalysisNotFound(
                    "no such analysis in this session", analysis_id=payload.analysis_id
                )
            candidates = await uow.observations.list_for_analysis(snapshot.id)
        analysis_id = snapshot.id

        matches = [
            item
            for item in candidates
            if item.kind is ObservationKind.ATTRIBUTION
            and (
                item.provenance.get("explanation_id") == payload.explanation_id
                if payload.explanation_id
                else (
                    item.model_projection.get("endpoint") == payload.endpoint
                    and item.model_projection.get("task") == payload.task
                )
            )
        ]
        if not matches:
            raise AnalysisNotFound(
                "no stored explanation matches this request; call "
                "get_or_create_explanation first",
                analysis_id=payload.analysis_id,
            )
        observation = matches[-1]
        package = package_from_observation(observation, top_k=payload.top_k)
        view = _package_view(package, analysis_id=analysis_id)
        return ToolOutput(
            canonical=observation.canonical_payload,
            model_view=view,
            ui_view=package.to_dict(),
            observation_ids=(observation.id,),
            provenance=observation.provenance,
        )

    return [
        ToolDefinition(
            name="get_or_create_explanation",
            title="Explain one endpoint, with a figure",
            description=(
                "Compute (or reuse) the atom-level explanation for exactly one endpoint, and "
                "for Tox21 exactly one assay. Returns an explanation_id, the observation_id "
                "every explanation claim must cite, a figure_id for the stored structure "
                "diagram, and the ranked positive and negative contributors. The figure and "
                "the contributor lists come from the same computation, so they always describe "
                "the same thing. Attribution shows what moved the model's score; it is not "
                "evidence of a chemical mechanism, and every claim written from it carries the "
                "attribution_not_causality limitation. When the response reports a non-null "
                "unmapped_importance, say so in the report: it is the share of the attribution "
                "that landed on no atom, and omitting it overstates how much of the score the "
                "picture explains. A failed explanation is a gap to record, never a target to "
                "silently drop."
            ),
            input_model=GetOrCreateExplanationInput,
            handler=get_or_create,
            profiles=frozenset({"report_build"}),
            soft_timeout_s=90.0,
            hard_timeout_s=180.0,
        ),
        ToolDefinition(
            name="get_explanation_package",
            title="Re-read an explanation already computed",
            description=(
                "Return a stored explanation package — figure reference, ranked contributors, "
                "unmapped importance and the observation id to cite — without asking the "
                "predictor for anything. Use it while drafting; use get_or_create_explanation "
                "to produce one that does not exist yet."
            ),
            input_model=GetExplanationPackageInput,
            handler=get_package,
            profiles=frozenset({"report_build"}),
            soft_timeout_s=3.0,
            hard_timeout_s=8.0,
        ),
    ]
