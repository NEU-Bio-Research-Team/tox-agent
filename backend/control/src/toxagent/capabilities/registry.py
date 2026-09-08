"""Semantic capabilities visible to planners.

Low-level tool sequences remain product-owned and cannot be expanded by a
runtime or by model output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..domain.investigation import GoalType, InvestigationPlan


#: Which observation kind each declared capability output arrives as.
#:
#: I31: a step's `success_condition` was free text that nothing evaluated, so
#: a step that produced nothing at all was marked completed and its question
#: marked answered. Coverage then read as sufficient on no evidence. A
#: capability already declares what it produces; this makes that declaration
#: checkable without asking a model to grade itself.
OUTPUT_OBSERVATION_KINDS: Mapping[str, str] = {
    "prediction_observation": "prediction",
    "attribution_observation": "attribution",
    "evidence_record": "evidence_record",
}


@dataclass(frozen=True, slots=True)
class CapabilityDefinition:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    tool_sequence: tuple[str, ...]
    allowed_goals: frozenset[GoalType]
    estimated_cost: int
    risk: str = "low"

    @property
    def expected_observation_kinds(self) -> frozenset[str]:
        return frozenset(
            OUTPUT_OBSERVATION_KINDS[name]
            for name in self.outputs
            if name in OUTPUT_OBSERVATION_KINDS
        )

    def satisfied_by(self, observations) -> bool:
        """Did this step actually produce what the capability promises?

        Deliberately a floor, not a grade: at least one observation of a kind
        this capability declares it outputs. It cannot judge whether an answer
        is good, and does not pretend to — what it rules out is the case the
        kernel used to accept, where nothing came back and the question was
        recorded as answered anyway.
        """
        expected = self.expected_observation_kinds
        if not expected:
            # A capability whose outputs are not observation-shaped: fall back
            # to "it produced something", which is still stronger than the
            # unconditional completion this replaces.
            return bool(observations)
        return any(observation.kind.value in expected for observation in observations)


class PlanViolation(ValueError):
    def __init__(self, code: str, detail: str, *, step_id: str | None = None) -> None:
        super().__init__(detail)
        self.code, self.detail, self.step_id = code, detail, step_id


class CapabilityRegistry:
    def __init__(self, definitions: tuple[CapabilityDefinition, ...]) -> None:
        names = [definition.name for definition in definitions]
        if len(names) != len(set(names)):
            raise ValueError("duplicate semantic capability")
        self._definitions = {definition.name: definition for definition in definitions}

    def get(self, name: str) -> CapabilityDefinition:
        try:
            return self._definitions[name]
        except KeyError:
            raise PlanViolation("unknown_capability", f"unknown capability {name!r}") from None

    def validate_plan(self, plan: InvestigationPlan, *, goal: GoalType,
                      supported_endpoints: frozenset[str], max_steps: int = 12,
                      max_replans: int = 2) -> None:
        if len(plan.steps) > max_steps:
            raise PlanViolation("unbounded_plan", f"plan has {len(plan.steps)} steps; max is {max_steps}")
        if plan.revision > max_replans + 1:
            raise PlanViolation("replan_budget", "replan revision exceeds the configured budget")
        signatures: set[tuple[str, tuple[str, ...]]] = set()
        forbidden = {"shell", "code_execution", "filesystem", "clinical_recommendation"}
        for step in plan.steps:
            definition = self.get(step.capability)
            if goal not in definition.allowed_goals:
                raise PlanViolation("goal_not_allowed", f"{step.capability!r} is not allowed for {goal.value}", step_id=step.id)
            if forbidden & set(definition.tool_sequence):
                raise PlanViolation("forbidden_tool", "capability expands to a forbidden tool", step_id=step.id)
            signature = (step.capability, step.input_refs)
            if signature in signatures:
                raise PlanViolation("duplicate_step", "duplicate capability/input step", step_id=step.id)
            signatures.add(signature)
            endpoint_refs = {ref.split(":", 1)[1] for ref in step.input_refs if ref.startswith("endpoint:")}
            unsupported = endpoint_refs - supported_endpoints
            if unsupported:
                raise PlanViolation("unsupported_endpoint", f"unsupported endpoints: {sorted(unsupported)}", step_id=step.id)

    def planner_view(self) -> tuple[Mapping[str, object], ...]:
        return tuple({
            "name": item.name, "inputs": item.inputs, "outputs": item.outputs,
            "allowed_goals": tuple(sorted(goal.value for goal in item.allowed_goals)),
            "estimated_cost": item.estimated_cost, "risk": item.risk,
        } for item in self._definitions.values())


def default_capabilities() -> CapabilityRegistry:
    all_prediction = frozenset({
        GoalType.EXPLAIN_PREDICTION, GoalType.COMPARE_ENDPOINTS,
        GoalType.INVESTIGATE_MODEL_BEHAVIOR, GoalType.PLAN_VERIFICATION,
    })
    evidence = frozenset({
        GoalType.ASSESS_EVIDENCE, GoalType.FIND_CONTRADICTIONS, GoalType.PLAN_VERIFICATION,
    })
    return CapabilityRegistry((
        CapabilityDefinition(
            "inspect_prediction", ("analysis_ref",), ("prediction_observation",),
            ("get_analysis_bundle", "get_analysis_slice"), all_prediction, 1,
        ),
        CapabilityDefinition(
            "inspect_attribution", ("analysis_ref", "endpoint"), ("attribution_observation",),
            ("get_attribution", "get_explanation_slice"),
            frozenset({GoalType.EXPLAIN_PREDICTION, GoalType.INVESTIGATE_MODEL_BEHAVIOR}), 2,
        ),
        CapabilityDefinition(
            "gather_external_evidence", ("query",), ("evidence_record",),
            ("search_toxicology_evidence", "get_evidence_record"), evidence, 4, "medium",
        ),
    ))
