"""``record_decision_plan``: the model proposes, the server issues and keeps.

Decision support has no fixed workflow, which is the point (ADR 0010). What it
lacked was any record of what the run set out to establish, so neither a
resumed turn nor a grader could tell a run that stopped because it had enough
from one that ran out. This tool lets the model write that down — a handful of
propositions and the kinds of source each needs — and nothing else. The server
validates the proposal, issues the ids and stores it in the run's
DecisionSupportStateV1. It does not decide what the model does next.

Registered only while the ``decision_state_plan_tool`` rollout flag is on, so
the default tool surface (and its schema hash) is unchanged until the flag is
canaried against the paired eval.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ...application import decision_state_service
from ...domain import decision_state as ds
from ...domain.errors import InvalidRequest
from ..registry import ToolContext, ToolDefinition, ToolOutput

TOOL_NAME = "record_decision_plan"


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PropositionInput(_Input):
    question: str = Field(
        min_length=1, max_length=500,
        description="One claim this answer needs to resolve, e.g. 'Does the compound block hERG "
                    "at therapeutically relevant exposure?'",
    )
    required_sources: list[Literal["prediction", "explanation", "evidence", "report"]] = Field(
        min_length=1, max_length=4,
        description="Which kinds of source could resolve it.",
    )


class RecordDecisionPlanInput(_Input):
    propositions: list[PropositionInput] = Field(min_length=1, max_length=ds.MAX_PROPOSITIONS)


def build(database) -> list[ToolDefinition]:
    async def record_plan(context: ToolContext, payload: RecordDecisionPlanInput) -> ToolOutput:
        proposed = [p.model_dump() for p in payload.propositions]
        async with database.unit_of_work() as uow:
            try:
                state = await decision_state_service.advance_in(
                    uow, context.run_id, lambda s: ds.apply_plan(s, proposed)
                )
            except ds.InvalidPlan as exc:
                raise InvalidRequest(str(exc)) from None
            if state is None:
                raise InvalidRequest("this run keeps no decision-support state")
            await uow.commit()
        view = {
            "revision": state.revision,
            "propositions": [
                {"id": p.id, "question": p.question, "status": p.status,
                 "required_sources": list(p.required_sources)}
                for p in state.propositions
            ],
            "note": (
                "Plan recorded. Resolve propositions by citing their sources in "
                "submit_grounded_answer's evidence_relations, using the same proposition text."
            ),
        }
        return ToolOutput(canonical=state.to_dict(), model_view=view, ui_view=view)

    return [
        ToolDefinition(
            name=TOOL_NAME,
            title="Record what this answer needs to establish",
            description=(
                "Optionally, early in the turn, record the few propositions your answer must "
                "resolve and which kinds of source could resolve each. The server issues ids "
                "and tracks coverage; calling it again replaces the propositions still open. "
                "It does not fetch anything and costs no evidence budget."
            ),
            input_model=RecordDecisionPlanInput,
            handler=record_plan,
            profiles=frozenset({"decision_support"}),
            soft_timeout_s=3.0,
            hard_timeout_s=10.0,
            idempotent=False,
            cost_class="cheap",
        )
    ]
