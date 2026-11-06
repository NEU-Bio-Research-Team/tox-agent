"""``plan_next_action``: one step of the gap-driven loop (#70, #71).

Before it searches, reads or answers, the investigator states the candidate
actions it is weighing; each names the requirement it serves, the gap or
competing explanation its result could tell apart, and one public sentence of
why. The server refuses what is inadmissible or repeated, ranks the rest in
§6.4's order, says when a counter-evidence check is due, and records the
choice in the run's gap ledger. It fetches nothing and costs no evidence
budget; the chosen action is then taken with its own tool.

Registered only while ``decision_loop_v1`` is on.
"""
from __future__ import annotations

import copy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ...application.investigation import decision_loop as loop
from ...application.investigation import decision_state_service
from ...domain import decision_state as ds
from ...domain.errors import InvalidRequest
from ...domain.investigation_gap import Gap, is_blocking
from ..registry import ToolContext, ToolDefinition, ToolOutput

TOOL_NAME = "plan_next_action"

_GAPS = tuple(g.value for g in Gap)
_EXPLANATIONS = tuple(e.value for e in loop.Explanation)


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CandidateInput(_Input):
    action: str = Field(min_length=1, max_length=64,
                        description="A tool you may call, or answer / ask_user / stop.")
    target: str = Field(default="", max_length=300,
                        description="The query, record id or procedure the action would run on.")
    requirement_id: str = Field(default="", max_length=40,
                                description="The goal contract question it serves (q1, ...).")
    gap: Literal[_GAPS] | Literal[""] = ""  # type: ignore[valid-type]
    discriminates: list[str] = Field(
        default_factory=list, max_length=4,
        description="Gaps or competing explanations whose answer this action's result could "
                    f"change. Gaps: {', '.join(_GAPS)}. Explanations: {', '.join(_EXPLANATIONS)}.",
    )
    source: Literal["observation", "chembl", "full_text", "europepmc", "pubchem", "unknown"] = "unknown"
    reuses: str = Field(default="", max_length=80,
                        description="An observation or evidence id already held that this reuses.")
    cost: int = Field(default=1, ge=0, le=20, description="Tool calls it would spend.")
    reason: str = Field(min_length=1, max_length=240, description="One public sentence: why this.")


class GapUpdateInput(_Input):
    requirement_id: str = Field(min_length=1, max_length=40)
    status: Literal["resolved", "open"]
    gap: Literal[_GAPS] | Literal[""] = ""  # type: ignore[valid-type]
    refs: list[str] = Field(default_factory=list, max_length=8)


class ExplanationInput(_Input):
    explanation: Literal[_EXPLANATIONS]  # type: ignore[valid-type]
    about: str = Field(min_length=1, max_length=200)


class PlanNextActionInput(_Input):
    candidates: list[CandidateInput] = Field(min_length=1, max_length=5)
    gap_updates: list[GapUpdateInput] = Field(default_factory=list, max_length=8)
    explanations: list[ExplanationInput] = Field(default_factory=list, max_length=4)


def build(database) -> list[ToolDefinition]:
    async def plan(context: ToolContext, payload: PlanNextActionInput) -> ToolOutput:
        async with database.unit_of_work() as uow:
            state = await uow.decision_states.get(context.run_id)
        if state is None:
            raise InvalidRequest("this run keeps no decision state")
        ledger = (copy.deepcopy(dict(state.decision_loop)) if state.decision_loop
                  else loop.seed_ledger(state.goal_contract))
        loop.resolve(ledger, [u.model_dump() for u in payload.gap_updates])
        for item in payload.explanations:
            entry = item.model_dump()
            if entry not in ledger["explanations"]:
                ledger["explanations"].append(entry)

        candidates = [loop.Candidate.from_dict(c.model_dump()) for c in payload.candidates]
        admissible = list(state.goal_contract.get("admissible_actions") or ())
        refused = loop.refusals(candidates, ledger=ledger, admissible=admissible)
        budget = state.budget_snapshot.get("max_tool_calls")
        budget_left = None if budget is None else max(0, budget - state.usage.get("tool_calls", 0))
        ranked = loop.rank(
            [c for i, c in enumerate(candidates) if i not in refused], ledger=ledger,
            access_failures=state.access_failures, budget_left=budget_left,
        )
        # rank() numbers the admissible subset; map back to the caller's order.
        kept = [i for i in range(len(candidates)) if i not in refused]
        for entry in ranked:
            entry["index"] = kept[entry["index"]]

        chosen = None
        note = ""
        if ranked:
            top = candidates[ranked[0]["index"]]
            due = loop.counter_evidence_due(
                ledger, top, search_allowed="search_toxicology_evidence" in admissible,
                budget_left=budget_left,
            )
            if due:
                note = due
            else:
                chosen = ranked[0]
                loop.mark_counter_evidence(ledger, top)
                ledger["actions"].append({
                    "step": len(ledger["actions"]) + 1, "action": top.action,
                    "target": top.target, "requirement_id": top.requirement_id,
                    "gap": top.gap, "discriminates": list(top.discriminates),
                    "reason": top.reason,
                })
                if top.gap and is_blocking(top.gap) and top.action not in {"ask_user", "stop"}:
                    note = ("this gap cannot be closed by more work in this run: name what is "
                            "missing and stop (needs_user_input or provider_failure)")
        await decision_state_service.advance(
            database, context.run_id, lambda s: ds.record_decision_loop(s, ledger),
        )
        view = {
            "chosen": chosen,
            "ranked": ranked,
            "refused": [{"index": i, "action": candidates[i].action, "why": why}
                        for i, why in sorted(refused.items())],
            "open_gaps": loop.with_procedures(
                loop.open_gaps(ledger)[:4],
                available="run_scientific_procedure" in admissible,
            ),
            "note": note or ("take the chosen action with its own tool; call plan_next_action "
                             "again before the next one"),
            "ranking_basis": "§6.4 order; carries_data_estimate is a dev-set prior, not a "
                             "learned probability",
        }
        if not ranked and not refused:
            raise InvalidRequest("give at least one candidate action")
        return ToolOutput(canonical={"ledger": ledger, **view}, model_view=view, ui_view=view)

    return [
        ToolDefinition(
            name=TOOL_NAME,
            title="Choose the next action by the gap it closes",
            description=(
                "Before each search, read or answer: list the candidate actions you are weighing "
                "(1-5). For each give the requirement it serves, the gap or competing "
                "explanation its result could tell apart (discriminates), the source, any "
                "observation it reuses, and one sentence of why. Report gaps you have resolved "
                "in gap_updates. The server refuses repeats and inadmissible actions, ranks the "
                "rest (identity/scope blockers first, then open decisive questions, reachable "
                "sources likely to hold data, reuse before new search, cost), tells you when one "
                "counter-evidence search is due, and records your choice. It fetches nothing."
            ),
            input_model=PlanNextActionInput,
            handler=plan,
            profiles=frozenset({"decision_support", "evidence_research"}),
            soft_timeout_s=3.0,
            hard_timeout_s=10.0,
            idempotent=False,
            cost_class="cheap",
        )
    ]
