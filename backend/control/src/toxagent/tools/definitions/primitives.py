"""Scientific primitives a skill cannot supply (RETHINK §4.7, §4.10, W9-13).

* ``compute_exposure_margin`` — IC50 over free Cmax, computed by the server from
  concentrations the session's sources state, stored as an observation a claim
  can cite.
* ``get_chembl_activities`` — measured activities of the analysed structure
  against a declared target, stored as citable evidence records.

Both behind ``scientific_primitives_v1`` (FLAG_GATED_TOOLS); the ChEMBL tool
also needs a configured provider, and both honour the researcher's data scope
(W9-07) — a compound restricted to internal data is never sent to ChEMBL.
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...application import scientific_case_service
from ...domain import exposure_margin as em
from ...domain import scientific_case as sc
from ...domain.errors import AnalysisNotFound, InvalidRequest, ToolDenied
from ...domain.events import EventType
from ...domain.evidence import EvidenceStatus
from ...domain.observation import Observation, ObservationKind, Producer
from ...research.normalization import hit_to_evidence
from ...research.policy import decide_acceptance
from ...research.relevance import RELEVANCE_POLICY_VERSION
from ..registry import ToolContext, ToolDefinition, ToolOutput

MARGIN_TOOL: Final[str] = "compute_exposure_margin"
CHEMBL_TOOL: Final[str] = "get_chembl_activities"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConcentrationInput(_Input):
    value: float = Field(gt=0)
    unit: Literal["pM", "nM", "µM", "uM", "mM", "M"]
    source_ref: str = Field(
        max_length=120,
        description="Where the number is written: 'context:<id>' (researcher-supplied) or "
                    "'evidence:<id>' (an accepted record). The value must appear there as written.",
    )


class FractionUnboundInput(_Input):
    value: float = Field(gt=0, le=1)
    source_ref: str = Field(max_length=120)


class ExposureMarginInput(_Input):
    ic50: ConcentrationInput
    cmax: ConcentrationInput = Field(
        description="Free Cmax, or total Cmax together with fraction_unbound.",
    )
    fraction_unbound: FractionUnboundInput | None = None


class ChemblActivitiesInput(_Input):
    analysis_id: str
    target: Literal["herg"] = "herg"
    limit: int = Field(default=10, ge=1, le=20)


async def _source_text(uow, context: ToolContext, ref: str) -> str:
    try:
        kind, identifier = sc.parse_ref(ref)
    except sc.InvalidCaseUpdate as exc:
        raise InvalidRequest(str(exc)) from None
    if kind == "context":
        case = await scientific_case_service.case_for_run(
            uow, session_id=context.session_id, run_id=context.run_id
        )
        item = next((c for c in case.context if c.id == identifier), None) if case else None
        if item is None:
            raise InvalidRequest(f"{ref}: this run's case has no such context item")
        return f"{item.key} {item.value} {item.note}"
    if kind == "evidence":
        record = await uow.evidence.get(identifier, session_id=context.session_id)
        if record is None or record.status is not EvidenceStatus.ACCEPTED:
            raise InvalidRequest(f"{ref}: not an accepted evidence record of this session")
        return " ".join((record.title, record.abstract_or_excerpt or "",
                         json.dumps(record.normalized_facts, ensure_ascii=False, default=str)))
    raise InvalidRequest(f"{ref}: a concentration cites 'context:' or 'evidence:', not {kind!r}")


def build(database, chembl_provider=None) -> list[ToolDefinition]:
    async def margin(context: ToolContext, payload: ExposureMarginInput) -> ToolOutput:
        inputs = [("ic50", payload.ic50.value, payload.ic50.source_ref),
                  ("cmax", payload.cmax.value, payload.cmax.source_ref)]
        if payload.fraction_unbound is not None:
            inputs.append(("fraction_unbound", payload.fraction_unbound.value,
                           payload.fraction_unbound.source_ref))
        async with database.unit_of_work() as uow:
            for name, value, ref in inputs:
                text = await _source_text(uow, context, ref)
                if not em.transcription_check(value, text):
                    raise InvalidRequest(
                        f"{name}: {value} is not written in {ref}. Give the number exactly as the "
                        "source states it; a value from memory is not accepted"
                    )
        try:
            result = em.compute(
                em.Concentration(payload.ic50.value, payload.ic50.unit, payload.ic50.source_ref),
                em.Concentration(payload.cmax.value, payload.cmax.unit, payload.cmax.source_ref),
                fraction_unbound=payload.fraction_unbound.value if payload.fraction_unbound else None,
                fu_source_ref=payload.fraction_unbound.source_ref if payload.fraction_unbound else None,
            )
        except em.InvalidMarginInput as exc:
            raise InvalidRequest(str(exc)) from None
        observation = Observation.create(
            session_id=context.session_id, run_id=context.run_id, producer=Producer.CALCULATOR,
            kind=ObservationKind.CALCULATION, schema_version=em.METHOD_VERSION,
            canonical_payload=result, model_projection={
                "calculation": "exposure_margin", **{k: result[k] for k in
                ("margin", "ic50_nM", "free_cmax_nM", "formula", "reading")},
                "field_paths": {"margin": "margin", "ic50_nM": "ic50_nM",
                                "free_cmax_nM": "free_cmax_nM"},
            },
            provenance={"method_version": em.METHOD_VERSION,
                        "source_refs": [ref for _, _, ref in inputs]},
            now=_now(), required_limitations=("screening_not_safety_assessment",),
        )
        async with database.unit_of_work() as uow:
            await uow.observations.add(observation)
            await uow.commit()
        return ToolOutput(canonical=result, model_view=observation.model_projection,
                          ui_view=result, observation_ids=(observation.id,),
                          provenance=observation.provenance)

    async def chembl(context: ToolContext, payload: ChemblActivitiesInput) -> ToolOutput:
        async with database.unit_of_work() as uow:
            snapshot = await uow.analyses.get(payload.analysis_id, session_id=context.session_id)
            refusal = (
                await scientific_case_service.external_search_refusal(
                    uow, session_id=context.session_id, analysis_id=payload.analysis_id,
                ) if snapshot is not None else None
            )
        if snapshot is None:
            raise AnalysisNotFound("no such analysis in this session", analysis_id=payload.analysis_id)
        if refusal is not None:
            raise ToolDenied(
                f"the researcher's data scope for this compound does not allow external lookups "
                f"({refusal}); ChEMBL was not queried.", reason="case_data_scope",
            )
        lookup = await chembl_provider.activities(
            canonical_smiles=snapshot.canonical_smiles, target=payload.target, limit=payload.limit,
        )
        if lookup.molecule_chembl_id is None:
            view = {"found": False, "target_chembl_id": lookup.target_chembl_id, "records": [],
                    "note": "ChEMBL has no record matching this structure. That is a "
                            "coverage gap, not evidence that it is inactive."}
            return ToolOutput(canonical=view, model_view=view, ui_view=view,
                              provenance={"analysis_id": snapshot.id, "provider": "chembl"})
        retrieved_at = _now()
        records: list[dict] = []
        ids: list[str] = []
        async with database.unit_of_work() as uow:
            for hit in lookup.hits:
                candidate = hit_to_evidence(hit, provider="chembl", session_id=context.session_id,
                                            retrieved_at=retrieved_at)
                existing = await uow.evidence.find_by_dedupe_key(context.session_id,
                                                                 candidate.dedupe_key)
                if existing is None:
                    final = decide_acceptance(candidate, allowed_hosts=chembl_provider.allowed_hosts)
                    # Matched by structure and by target, not by words.
                    final = replace(final, relevance_assessment={
                        "relevance": "direct", "reason_codes": ["structure_match", "target_match"],
                        "policy_version": RELEVANCE_POLICY_VERSION,
                    })
                    await uow.evidence.add(final)
                    uow.emit(session_id=context.session_id, type=EventType.EVIDENCE_CREATED,
                             entity_type="evidence", entity_id=final.id, run_id=context.run_id,
                             payload={"provider": "chembl", "status": final.status.value})
                else:
                    final = existing
                if final.status is EvidenceStatus.ACCEPTED:
                    ids.append(final.id)
                    records.append({"evidence_id": final.id, "title": final.title,
                                    "excerpt": final.abstract_or_excerpt,
                                    "facts": final.normalized_facts})
            await uow.commit()
        view = {
            "found": True, "molecule_chembl_id": lookup.molecule_chembl_id,
            "structure_matches": lookup.structure_matches,
            "target_chembl_id": lookup.target_chembl_id, "records": records,
            "note": "Measured values from ChEMBL, each under its own assay conditions; compare "
                    "them only with their assay type and units. Cite a record by evidence_id.",
        }
        return ToolOutput(canonical=view, model_view=view, ui_view=view,
                          observation_ids=tuple(ids),
                          provenance={"analysis_id": snapshot.id, "provider": "chembl",
                                      "molecule_chembl_id": lookup.molecule_chembl_id})

    tools = [
        ToolDefinition(
            name=MARGIN_TOOL,
            title="Compute an exposure margin",
            description=(
                "IC50 divided by free Cmax, computed by the server from concentrations written in "
                "a context item or an accepted evidence record (the number must appear there as "
                "written). Returns an observation whose margin, ic50_nM and free_cmax_nM fields a "
                "numeric claim can cite. It is a ratio of two measurements, never a safety verdict."
            ),
            input_model=ExposureMarginInput, handler=margin,
            profiles=frozenset({"decision_support"}),
            soft_timeout_s=3.0, hard_timeout_s=10.0, cost_class="cheap",
        ),
    ]
    if chembl_provider is not None:
        tools.append(ToolDefinition(
            name=CHEMBL_TOOL,
            title="Look up measured activities in ChEMBL",
            description=(
                "Measured IC50/Ki/Kd/EC50 values for the analysed structure (matched by structure, "
                "which can include stereoisomers; the number of matches is reported) against a "
                "declared target, from ChEMBL. Each becomes an evidence record you can cite by "
                "evidence_id; the records are returned whole."
            ),
            input_model=ChemblActivitiesInput, handler=chembl,
            profiles=frozenset({"decision_support"}),
            soft_timeout_s=15.0, hard_timeout_s=40.0, max_retries=1, cost_class="moderate",
        ))
    return tools
