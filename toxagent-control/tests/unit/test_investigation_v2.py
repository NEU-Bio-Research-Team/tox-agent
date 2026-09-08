from datetime import datetime, timezone

import pytest

from toxagent.agent.budget import PROFILES, BudgetUsage, StopReason, stop_reason
from toxagent.capabilities.registry import PlanViolation, default_capabilities
from toxagent.domain.ids import SESSION, new_id
from toxagent.domain.investigation import (
    CaseState, Coverage, GoalType, InvestigationPlan, InvestigationStep,
)


NOW = datetime.now(timezone.utc)


def make_case(goal=GoalType.EXPLAIN_PREDICTION):
    return CaseState.create(
        session_id=new_id(SESSION), subject={"smiles": "CCO"}, goal=goal,
        questions=("What supports the label?",), now=NOW,
    )


def test_case_revision_requires_reason_and_is_reconstructable():
    case = make_case()
    revised = case.revise(reason="analysis_attached", now=NOW, active_analysis_id=None)
    assert revised.revision == 2
    assert revised.to_dict()["goal"] == "explain_prediction"
    assert "chain_of_thought" not in revised.to_dict()
    with pytest.raises(ValueError):
        case.revise(reason="", now=NOW)


def test_coverage_and_budget_stopping_are_deterministic():
    coverage = Coverage(("q1", "q2"), ("q1",))
    assert coverage.ratio == 0.5 and not coverage.sufficient
    limits = PROFILES["report_qa"]
    assert stop_reason(limits, BudgetUsage(tool_calls=9), coverage_sufficient=False, elapsed_s=1) is StopReason.BUDGET_EXHAUSTED
    assert stop_reason(limits, BudgetUsage(), coverage_sufficient=True, elapsed_s=1) is StopReason.COVERAGE_COMPLETE


def test_plan_validation_rejects_unknown_duplicate_and_unsupported_endpoint():
    case = make_case()
    step = InvestigationStep.create(
        question="q", capability="inspect_prediction", input_refs=("endpoint:clintox",),
        expected_output="prediction", success_condition="observation exists", case_revision=1,
    )
    plan = InvestigationPlan.create(case_id=case.id, reason="initial", steps=(step,), now=NOW)
    with pytest.raises(PlanViolation, match="unsupported"):
        default_capabilities().validate_plan(plan, goal=case.goal, supported_endpoints=frozenset({"herg"}))
