"""``resolve_compound_record`` — substance identity through an approved provider.

Report spec sections 5.4 and 9. The tool takes *no* name and *no* free-text
query: it resolves the analysis snapshot's canonical SMILES, so the record
always describes the molecule that was actually predicted on. A model that
could pass a name here could steer identity resolution toward the compound it
expected, and the report would then carry a real provider citation for the
wrong substance — which is worse than no identity at all.

Provider selection is server policy (spec section 9). The input schema has no
provider, host or URL field, and there is no path by which one arrives.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ...application.investigation import scientific_case_service
from ...domain.errors import AnalysisNotFound, ToolDenied
from ...domain.events import EventType
from ...domain.observation import Observation, ObservationKind, Producer
from ...platform.flags import is_enabled
from ...research.compound import CompoundProvider
from ..registry import ToolContext, ToolDefinition, ToolOutput


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResolveCompoundInput(_Input):
    analysis_id: str = Field(
        description=(
            "The analysis whose canonical SMILES to resolve. Identity is looked up by "
            "structure; there is no way to look it up by name."
        )
    )


def build(database, provider: CompoundProvider) -> list[ToolDefinition]:
    async def resolve(context: ToolContext, payload: ResolveCompoundInput) -> ToolOutput:
        async with database.unit_of_work() as uow:
            snapshot = await uow.analyses.get(
                payload.analysis_id, session_id=context.session_id
            )
        if snapshot is None:
            raise AnalysisNotFound(
                "no such analysis in this session", analysis_id=payload.analysis_id
            )
        if is_enabled("scientific_case_v1"):
            # W9-07's data scope covers every lookup that sends the structure
            # out, not only literature search: a live report build on a case
            # restricted to internal data sent its SMILES here (2026-09-26).
            async with database.unit_of_work() as uow:
                refusal = await scientific_case_service.external_search_refusal(
                    uow, session_id=context.session_id, analysis_id=snapshot.id,
                )
            if refusal is not None:
                raise ToolDenied(
                    "the researcher's data scope for this compound does not allow external "
                    f"lookups ({refusal}); the chemical database was not queried. Describe "
                    "the compound by its analysed structure only.",
                    reason="case_data_scope",
                )

        record = await provider.resolve(canonical_smiles=snapshot.canonical_smiles)
        canonical = {**record.to_dict(), "raw": dict(record.raw)}
        view = record.model_view()
        view["analysis_id"] = snapshot.id

        # Persisted as an observation for the same reason a prediction is: a
        # report claim about a compound's name or molecular weight has to cite
        # something the validator can resolve, and "the provider said so during
        # the run" is not a citable thing after the run ends.
        observation = Observation.create(
            session_id=context.session_id,
            run_id=context.run_id,
            producer=Producer.RESEARCH,
            kind=ObservationKind.EVIDENCE_RECORD,
            schema_version="compound-record-v1",
            canonical_payload=canonical,
            model_projection=view,
            provenance={
                "analysis_id": snapshot.id,
                "provider": record.provider,
                "canonical_url": record.canonical_url,
                "retrieved_at": record.retrieved_at.isoformat(),
                "content_sha256": record.content_sha256,
                "resolved": record.resolved,
            },
            now=record.retrieved_at,
        )
        async with database.unit_of_work() as uow:
            await uow.observations.add(observation, analysis_id=snapshot.id)
            uow.emit(
                session_id=context.session_id, type=EventType.OBSERVATION_CREATED,
                entity_type="observation", entity_id=observation.id, run_id=context.run_id,
                payload={"kind": "compound_record", "resolved": record.resolved},
            )
            await uow.commit()

        return ToolOutput(
            canonical=canonical,
            model_view=observation.model_projection,
            ui_view=record.to_dict(),
            observation_ids=(observation.id,),
            provenance=observation.provenance,
        )

    return [
        ToolDefinition(
            name="resolve_compound_record",
            title="Resolve substance identity and properties",
            description=(
                "Look up the analysed structure in the configured chemical database and return "
                "its preferred name, synonyms, registry identifiers and selected bulk "
                "properties, each with the observation id and field path needed to cite it. "
                "The lookup is by canonical SMILES only. When the database has no record, the "
                "response says resolved=false and every identity field is null — report that "
                "as an identity gap. Never fill a null field from a compound that merely looks "
                "similar, and never present a name from this tool as a predictor result: these "
                "are external facts about the substance, not model output."
            ),
            input_model=ResolveCompoundInput,
            handler=resolve,
            profiles=frozenset({"report_build"}),
            soft_timeout_s=15.0,
            hard_timeout_s=35.0,
        )
    ]
