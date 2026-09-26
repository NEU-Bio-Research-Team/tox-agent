"""``get_scientific_case`` / ``update_scientific_case``: the model's hands on the case.

ADR 0012. The case is the investigation that outlives a turn: the decision
question, competing hypotheses, a ledger of evidence for and against, what is
still unknown, why the agent acted, and what it can conclude. These two tools
are the only way a model reads or changes it, and neither decides what the
model does next.

The update schema is one flat operation object rather than a tagged union:
every provider this product runs on handles a flat object with an ``op`` enum
reliably, and the domain names exactly which field an operation is missing.

Refs are checked here against what the session really holds — an observation
of the right kind, an accepted evidence record, a report artifact — before the
domain sees them, so a ledger entry can never cite something that does not
exist. Registered only while ``scientific_case_v1`` is on
(``tools.registry.FLAG_GATED_TOOLS``).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...application import scientific_case_service as service
from ...domain import scientific_case as sc
from ...domain.errors import Conflict, InvalidRequest
from ...domain.evidence import EvidenceStatus
from ...domain.observation import ObservationKind
from ..registry import ToolContext, ToolDefinition, ToolOutput

GET_TOOL = "get_scientific_case"
UPDATE_TOOL = "update_scientific_case"

#: How many ledger entries a read shows. The rest are counted, and the full
#: case is on the API for a human reader.
_EVIDENCE_WINDOW = 40


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScopeInput(_Input):
    endpoint: str | None = Field(default=None, max_length=200)
    species: str | None = Field(default=None, max_length=200)
    assay: str | None = Field(default=None, max_length=200)
    dose: str | None = Field(default=None, max_length=200)
    use_context: str | None = Field(default=None, max_length=200)
    compound: str | None = Field(default=None, max_length=200)


class ConclusionLineInput(_Input):
    text: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(
        min_length=1, max_length=8,
        description="Ledger entry ids (e1, e2…) this line rests on.",
    )


class CaseOperation(_Input):
    op: Literal[
        "set_question", "add_hypothesis", "revise_hypothesis", "record_evidence",
        "record_uncertainty", "resolve_uncertainty", "record_action", "propose_next_test",
        "set_conclusion",
    ]
    # set_question
    question: str | None = Field(default=None, max_length=2000,
                                 description="set_question: the decision the research supports.")
    decision_context: str | None = Field(default=None, max_length=1000)
    # add_hypothesis / revise_hypothesis
    statement: str | None = Field(default=None, max_length=500,
                                  description="add_hypothesis: one competing explanation.")
    hypothesis_kind: Literal[
        "model_signal", "mechanism", "assay_or_exposure", "data_insufficient",
        "alternative_explanation", "other",
    ] | None = None
    refutation_condition: str | None = Field(
        default=None, max_length=500,
        description="add_hypothesis: the observation that would refute it. Required.",
    )
    hypothesis_id: str | None = Field(default=None, description="revise_hypothesis: h1, h2…")
    status: Literal["open", "supported", "weakened", "refuted", "unresolvable"] | None = None
    reason: str | None = Field(default=None, max_length=500)
    # record_evidence
    claim: str | None = Field(default=None, max_length=1000,
                              description="record_evidence: what the source says, in one sentence.")
    source_class: Literal[
        "predictor_fact", "explanation_fact", "external_experimental", "external_regulatory",
        "report_fact", "user_supplied",
    ] | None = None
    source_ref: str | None = Field(
        default=None, max_length=120,
        description="record_evidence: 'observation:<id>', 'evidence:<id>', 'report:<id>' or "
                    "'context:<id>' — an artifact you read with a tool, or a context item.",
    )
    stance: Literal["supports", "contradicts", "contextual", "insufficient"] | None = None
    directness: Literal[
        "direct", "analogue", "class_level", "mechanistic_inference", "indirect", "not_assessed",
    ] | None = None
    locator: str | None = Field(default=None, max_length=300,
                                description="record_evidence: field path or quoted span.")
    scope: ScopeInput | None = None
    hypothesis_ids: list[str] = Field(default_factory=list, max_length=8)
    # record_uncertainty / resolve_uncertainty
    uncertainty_kind: Literal[
        "missing_endpoint", "model_calibration", "applicability_domain", "explainer_faithfulness",
        "ocr_ambiguity", "conflicting_sources", "assay_mismatch", "missing_exposure",
        "missing_data", "provider_coverage", "other",
    ] | None = None
    description: str | None = Field(default=None, max_length=1000)
    severity: Literal["low", "medium", "high", "blocking"] | None = None
    uncertainty_id: str | None = None
    resolution: str | None = Field(default=None, max_length=1000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    # record_action
    action: str | None = Field(
        default=None, max_length=120,
        description="record_action: what you did or will do, e.g. 'search_toxicology_evidence' "
                    "or 'counterevidence_search'.",
    )
    purpose: str | None = Field(default=None, max_length=500,
                                description="record_action: why — which hypothesis it could move.")
    decision: Literal["continue", "redirect", "ask_user", "answer", "stop"] | None = None
    outcome: str | None = Field(default=None, max_length=500)
    # propose_next_test
    test: str | None = Field(default=None, max_length=500)
    rationale: str | None = Field(default=None, max_length=1000)
    discriminates: list[str] = Field(
        default_factory=list, max_length=8,
        description="propose_next_test: hypothesis ids whose status the result would change.",
    )
    expected_readouts: list[str] = Field(default_factory=list, max_length=6)
    # set_conclusion
    can_say: list[ConclusionLineInput] = Field(default_factory=list, max_length=8)
    cannot_say: list[str] = Field(default_factory=list, max_length=8)
    what_would_change: list[str] = Field(default_factory=list, max_length=8)
    approver: str | None = Field(default=None, max_length=200)


class UpdateScientificCaseInput(_Input):
    #: One call per turn carries the whole update (W9-01), so the cap fits a
    #: first turn: two or three hypotheses, their evidence, an action, an
    #: uncertainty and the conclusion.
    operations: list[CaseOperation] = Field(min_length=1, max_length=16)


class GetScientificCaseInput(_Input):
    include_evidence: bool = Field(
        default=True, description="Include the most recent ledger entries.",
    )


def _payload(operation: CaseOperation) -> dict[str, Any]:
    """The tool's flat operation as the domain's payload for that op."""
    op = operation.op
    data = operation.model_dump(exclude_none=True)
    data.pop("op")
    if op == "add_hypothesis" and "hypothesis_kind" in data:
        data["kind"] = data.pop("hypothesis_kind")
    if op == "record_uncertainty" and "uncertainty_kind" in data:
        data["kind"] = data.pop("uncertainty_kind")
    if op == "set_conclusion":
        data["can_say"] = [line.model_dump() for line in operation.can_say]
    return data


async def _check_ref(uow, context: ToolContext, index: int, operation: CaseOperation) -> None:
    if operation.op != "record_evidence" or not operation.source_ref:
        return
    try:
        kind, identifier = sc.parse_ref(operation.source_ref)
    except sc.InvalidCaseUpdate as exc:
        raise InvalidRequest(f"operations[{index}]: {exc}") from None
    where = f"operations[{index}].source_ref"
    if kind == "observation":
        observation = await uow.observations.get(identifier, session_id=context.session_id)
        if observation is None:
            raise InvalidRequest(f"{where}: this session has no observation {identifier!r}")
        expected = {
            "predictor_fact": ObservationKind.PREDICTION,
            "explanation_fact": ObservationKind.ATTRIBUTION,
        }.get(operation.source_class or "")
        if expected is not None and observation.kind is not expected:
            raise InvalidRequest(
                f"{where}: {identifier} is a {observation.kind.value} observation, which is not a "
                f"{operation.source_class}"
            )
    elif kind == "evidence":
        record = await uow.evidence.get(identifier, session_id=context.session_id)
        if record is None or record.status is not EvidenceStatus.ACCEPTED:
            raise InvalidRequest(
                f"{where}: {identifier!r} is not an accepted evidence record of this session; "
                "cite a record returned by search_toxicology_evidence/get_evidence_record"
            )
    elif kind == "report":
        if await uow.reports.get_artifact(identifier, session_id=context.session_id) is None:
            raise InvalidRequest(f"{where}: this session has no report {identifier!r}")
    elif kind != "context":
        raise InvalidRequest(
            f"{where}: the ref kind must be observation, evidence, report or context; got {kind!r}"
        )


def _model_view(case: sc.ScientificCaseV1, *, include_evidence: bool = True) -> dict[str, Any]:
    full = case.to_dict()
    view: dict[str, Any] = {
        "case_id": full["case_id"], "revision": full["revision"], "question": full["question"],
        "decision_context": full["decision_context"], "subject_refs": full["subject_refs"],
        "context": full["context"], "hypotheses": full["hypotheses"],
        "uncertainties": full["uncertainties"], "next_tests": full["next_tests"],
        "conclusion": full["conclusion"], "coverage": full["coverage"],
        "runs": len(full["runs"]), "actions_recorded": len(full["actions"]),
    }
    if include_evidence:
        view["evidence"] = full["evidence"][-_EVIDENCE_WINDOW:]
        view["evidence_total"] = len(full["evidence"])
    return view


def _issued(before: sc.ScientificCaseV1, after: sc.ScientificCaseV1) -> dict[str, list[str]]:
    def new(attr: str) -> list[str]:
        old = {item.id for item in getattr(before, attr)}
        return [item.id for item in getattr(after, attr) if item.id not in old]

    return {
        "hypotheses": new("hypotheses"), "evidence": new("evidence"),
        "uncertainties": new("uncertainties"), "next_tests": new("next_tests"),
        "actions": new("actions"),
    }


def build(database) -> list[ToolDefinition]:
    async def get_case(context: ToolContext, payload: GetScientificCaseInput) -> ToolOutput:
        async with database.unit_of_work() as uow:
            case = await service.case_for_run(uow, session_id=context.session_id,
                                              run_id=context.run_id)
        if case is None:
            raise InvalidRequest("this run is not attached to a scientific case")
        view = _model_view(case, include_evidence=payload.include_evidence)
        return ToolOutput(canonical=case.to_dict(), model_view=view, ui_view=case.to_dict())

    async def update_case(context: ToolContext, payload: UpdateScientificCaseInput) -> ToolOutput:
        async with database.unit_of_work() as uow:
            before = await service.case_for_run(uow, session_id=context.session_id,
                                                run_id=context.run_id)
            if before is None:
                raise InvalidRequest("this run is not attached to a scientific case")
            for index, operation in enumerate(payload.operations):
                await _check_ref(uow, context, index, operation)
            updates = [
                service.update(operation.op, actor=sc.Actor.MODEL.value, run_id=context.run_id,
                               **_payload(operation))
                for operation in payload.operations
            ]
            try:
                after = await service.apply_updates(
                    uow, case_id=before.id, session_id=context.session_id, updates=updates,
                )
            except sc.InvalidCaseUpdate as exc:
                raise InvalidRequest(
                    f"no operation was applied: {exc}", case_revision=before.revision,
                ) from None
            except Conflict:
                raise InvalidRequest(
                    "the case changed while this update was being applied; read it again with "
                    "get_scientific_case and resend",
                ) from None
            await uow.commit()
        view = {
            "case_id": after.id, "revision": after.revision,
            "applied": [operation.op for operation in payload.operations],
            "issued_ids": _issued(before, after), "coverage": after.coverage,
        }
        return ToolOutput(canonical=after.to_dict(), model_view=view, ui_view=view)

    return [
        ToolDefinition(
            name=GET_TOOL,
            title="Read the scientific case",
            description=(
                "Read the investigation this turn belongs to: the decision question, context the "
                "researcher supplied, competing hypotheses with their status, the evidence ledger "
                "(for/against), open uncertainties, proposed tests and the current conclusion. "
                "It persists across turns. Costs no evidence budget."
            ),
            input_model=GetScientificCaseInput,
            handler=get_case,
            profiles=frozenset({"decision_support"}),
            soft_timeout_s=3.0,
            hard_timeout_s=10.0,
            cost_class="cheap",
        ),
        ToolDefinition(
            name=UPDATE_TOOL,
            title="Update the scientific case",
            description=(
                "Record what the investigation has established, as 1-10 operations applied "
                "all-or-nothing. op is one of: set_question; add_hypothesis (statement, "
                "hypothesis_kind, refutation_condition); revise_hypothesis (hypothesis_id, status, "
                "reason — supported/weakened/refuted need a ledger entry of that stance first); "
                "record_evidence (claim, source_class, source_ref of an artifact you read, stance, "
                "directness, hypothesis_ids, optional locator/scope); record_uncertainty "
                "(uncertainty_kind, description, severity, hypothesis_ids); resolve_uncertainty "
                "(uncertainty_id, resolution, evidence_ids); record_action (action, purpose, "
                "decision — why you are searching, reading, asking or stopping; use action "
                "'counterevidence_search' when you look for evidence against a hypothesis); "
                "propose_next_test (test, rationale, discriminates, expected_readouts); "
                "set_conclusion (can_say lines each citing ledger evidence_ids, cannot_say, "
                "what_would_change). Ids (h1, e1, u1, t1) are issued by the server and returned. "
                "A model score or a highlight is never independent evidence; an inference is a "
                "hypothesis, not evidence. This does not replace submit_grounded_answer."
            ),
            input_model=UpdateScientificCaseInput,
            handler=update_case,
            profiles=frozenset({"decision_support"}),
            soft_timeout_s=3.0,
            hard_timeout_s=10.0,
            idempotent=False,
            cost_class="cheap",
        ),
    ]
