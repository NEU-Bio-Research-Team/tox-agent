"""Scientific primitives a skill cannot supply (RETHINK §4.7, §4.10, W9-13).

* ``compute_exposure_margin`` — IC50 over free Cmax, computed by the server from
  concentrations the session's sources state, stored as an observation a claim
  can cite.
This calculation is behind ``scientific_primitives_v1`` (FLAG_GATED_TOOLS).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...application.investigation import scientific_case_service
from ...domain import exposure_margin as em
from ...domain import scientific_case as sc
from ...domain.errors import InvalidRequest, ToolDenied
from ...domain.evidence import EvidenceStatus
from ...domain.observation import Observation, ObservationKind, Producer
from ..registry import ToolContext, ToolDefinition, ToolOutput

MARGIN_TOOL: Final[str] = "compute_exposure_margin"


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
        # The full text this session captured counts as the source too: a
        # potency read off a paper's table row is as written as one in its
        # abstract. Found live on 2026-10-29 -- T5's IC50 is only in Table 1,
        # and the margin was refused three times while the model quoted the row.
        full_text = await uow.evidence_text.get(identifier, session_id=context.session_id)
        passages = " ".join(
            " ".join((passage.text, *passage.cells))
            for passage in (full_text.passages if full_text is not None else ())
        )
        return " ".join((record.title, record.abstract_or_excerpt or "",
                         json.dumps(record.normalized_facts, ensure_ascii=False, default=str),
                         passages))
    raise InvalidRequest(f"{ref}: a concentration cites 'context:' or 'evidence:', not {kind!r}")


#: What a margin needs and this server does not check. Written on the claim
#: because §5.3 asks for assumptions to be reviewable, not implied.
_MARGIN_ASSUMPTIONS: Final[tuple[str, ...]] = (
    "exposure_context_compatible",
    "potency_and_exposure_of_the_same_compound",
)


def _claim_input(case: sc.ScientificCaseV1, ref: str) -> str | None:
    """The ref a derived claim cites for one input of the calculation.

    A researcher value is cited by its version, not its context item: when the
    researcher corrects it, the case marks what cited the superseded version,
    and an item id would not say which version that was."""
    kind, identifier = sc.parse_ref(ref)
    if kind == "context":
        version = sc.version_for_item(case, identifier)
        return version.ref if version is not None else None
    return ref


async def _record_margin_claim(database, context: ToolContext, result: dict[str, Any],
                               refs: list[str]) -> str | None:
    """§11 turn 2: the margin as a derived claim over the input versions.

    The server records it, not the investigator, because the server did the
    division (§9.1; ``record_derived_claim`` is the server's op). Without a
    case there is nothing to record into, and the observation still stands.
    """
    statement = (f"exposure margin {result['margin']:g} = IC50 {result['ic50_nM']:g} nM / "
                 f"free Cmax {result['free_cmax_nM']:g} nM")

    def updates_for(case: sc.ScientificCaseV1) -> list[sc.CaseUpdate]:
        claim_inputs = [_claim_input(case, ref) for ref in refs]
        if None in claim_inputs:
            return []
        return [scientific_case_service.update(
            "record_derived_claim", actor=sc.Actor.SERVER.value, run_id=context.run_id,
            statement=statement, rule=em.METHOD_VERSION, inputs=claim_inputs,
            assumptions=list(_MARGIN_ASSUMPTIONS),
        )]

    case = await scientific_case_service.advance(
        database, session_id=context.session_id, run_id=context.run_id, updates_for=updates_for,
    )
    if case is None:
        return None
    return next((claim.id for claim in case.derived_claims
                 if claim.statement.casefold() == statement.casefold()), None)


async def _assay_subject(database, context, payload, binding) -> tuple[str | None, str]:
    """The structure an assay lookup is about, and the analysis it belongs to.

    AG-EVIDENCE: a screening subject is looked up by its own ligand -- but only
    when its target is the one this tool serves (the catalog maps KCNH2 to
    hERG). Any other target is a capability the deployment does not have; it
    is never a string handed to the hERG lookup.
    """
    from ...domain.errors import CapabilityUnavailable, InvalidRequest
    from ...domain.subject import SubjectKind
    from ...research import target_catalog

    if binding is not None and binding.kind is SubjectKind.ACTIVITY_SCREENING:
        toxicity = binding.artifact("toxicity_snapshot")
        local = toxicity.local_id if toxicity is not None else None
        if payload.analysis_id and payload.analysis_id != local:
            raise ToolDenied("this run is about another subject", reason="subject_mismatch")
        target = binding.identity.target
        entry = target_catalog.lookup(target.target_id if target else None,
                                      uniprot_id=target.uniprot_id if target else None)
        if entry is None or entry.toxicity_endpoint != payload.target:
            raise CapabilityUnavailable(
                f"Assay activities are served for {payload.target} only; this screening's "
                f"target is {target.target_id if target else 'unknown'}",
                capability="assay_activities_for_target",
                target_catalog=target_catalog.CATALOG_VERSION,
            )
        return payload.analysis_id or local, binding.identity.canonical_smiles
    if binding is not None and payload.analysis_id and payload.analysis_id != binding.analysis_id:
        raise ToolDenied("this run is about another subject", reason="subject_mismatch")
    analysis_id = payload.analysis_id or (binding.analysis_id if binding is not None else None)
    if not analysis_id:
        raise InvalidRequest("name the analysis to look up")
    return analysis_id, ""


def build(database) -> list[ToolDefinition]:
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
        claim_id = await _record_margin_claim(database, context, result,
                                              [ref for _, _, ref in inputs])
        view = dict(observation.model_projection)
        if claim_id is not None:
            view["derived_claim"] = claim_id
        return ToolOutput(canonical={**result, "derived_claim": claim_id}, model_view=view,
                          ui_view=result, observation_ids=(observation.id,),
                          provenance=observation.provenance)

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
    return tools
