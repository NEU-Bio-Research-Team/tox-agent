"""What the kernel is allowed to spend, and what it may claim it established.

I31 found four ways the kernel could report more than it had done. It is not
on the production path — the compatibility gateway still is — so these are the
gates it has to pass before a cutover can be considered, not a description of
what serves traffic today.

- The plan call happened before any budget check and recorded its turn
  afterwards, so `model_turns=1` bought a plan *and* a compose: two calls
  against a limit of one.
- Compose ran unconditionally and added a turn, whatever the budget said.
- A step that produced nothing was marked completed and its question marked
  answered. `success_condition` was free text nothing evaluated, so coverage
  read as sufficient on no evidence.
- The stop reason was recomputed after the loop, so a terminal
  `budget_exhausted` could be overwritten by `coverage_complete`.
- `COMPLETED` was written before the answer validators had run.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from toxagent.agent.budget import BudgetExhausted, BudgetLimits, StopReason
from toxagent.agent.kernel import KernelState, ScientificAgentKernel
from toxagent.answer.compiler import AnswerCompiler, SemanticAnswerDraft
from toxagent.capabilities.registry import default_capabilities
from toxagent.domain.ids import RUN, SESSION, new_id
from toxagent.domain.investigation import (
    CaseState, Coverage, GoalType, InvestigationPlan, InvestigationStep, StepStatus,
)
from toxagent.domain.observation import Observation, ObservationKind, Producer

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)
LIMITS = BudgetLimits(
    model_turns=4, tool_calls=8, searches=0, reads=0, attributions=2, replans=0,
    wall_time_s=120,
)


class _Repository:
    """In-memory case store with the durability the kernel actually relies on."""

    def __init__(self) -> None:
        self.cases: dict[str, CaseState] = {}
        self.plans: dict[str, InvestigationPlan] = {}
        self.steps: dict[str, InvestigationStep] = {}
        self.observations: dict[str, Observation] = {}
        self.transitions: list[KernelState] = []

    async def save_case(self, case, *, expected_revision):
        self.cases[case.id] = case

    async def save_plan(self, plan):
        self.plans[plan.id] = plan
        for step in plan.steps:
            self.steps[step.id] = step

    async def save_step(self, plan_id, step):
        self.steps[step.id] = step
        plan = self.plans[plan_id]
        self.plans[plan_id] = InvestigationPlan(
            plan.id, plan.case_id, plan.revision, plan.reason,
            tuple(self.steps[s.id] for s in plan.steps), plan.created_at,
        )

    async def append_transition(self, case_id, transition):
        self.transitions.append(transition.state)

    async def get_plan(self, plan_id):
        return self.plans.get(plan_id)

    async def load_observations(self, ids):
        return tuple(self.observations[i] for i in ids if i in self.observations)


class _Planner:
    """Counts model calls, which is the thing the budget is about."""

    def __init__(self, case_id: str, *, steps: int = 1) -> None:
        self.plan_calls = 0
        self.compose_calls = 0
        self._case_id = case_id
        self._steps = steps

    async def plan(self, case, capabilities):
        self.plan_calls += 1
        return InvestigationPlan.create(
            case_id=case.id, reason="initial", now=NOW,
            steps=tuple(
                InvestigationStep.create(
                    question=f"question {n}", capability="inspect_prediction",
                    # Distinct inputs: the plan validator rejects two steps
                    # with the same capability and the same refs.
                    input_refs=(f"analysis_ref:ana_{n}",), expected_output="prediction",
                    success_condition="a prediction observation exists", case_revision=1,
                )
                for n in range(self._steps)
            ),
        )

    async def compose(self, case, observations):
        self.compose_calls += 1
        return SemanticAnswerDraft(sections=(("Summary", ()),))


def _case(questions=("question 0",)) -> CaseState:
    return CaseState.create(
        session_id=new_id(SESSION), subject={"smiles": "CCO"},
        goal=GoalType.EXPLAIN_PREDICTION, questions=questions, now=NOW,
    )


def _observation(session_id: str) -> Observation:
    return Observation.create(
        session_id=session_id, run_id=new_id(RUN), producer=Producer.PREDICTOR,
        kind=ObservationKind.PREDICTION, schema_version="prediction-v1",
        canonical_payload={"probability": 0.73}, model_projection={"probability": 0.73},
        provenance={"model_id": "herg-tox21-chemberta-v1"}, now=NOW,
    )


def _kernel(repository, planner, executor):
    return ScientificAgentKernel(
        capabilities=default_capabilities(), repository=repository, planner=planner,
        execute_capability=executor, compiler=AnswerCompiler(), clock=lambda: NOW,
    )


async def _run(kernel, case, limits=LIMITS, elapsed=lambda: 0.0):
    return await kernel.run(
        case, supported_endpoints=frozenset({"herg", "tox21"}), limits=limits, elapsed=elapsed,
    )


async def test_one_model_turn_does_not_buy_two_model_calls():
    """The headline: plan and compose are both model calls."""
    case = _case()
    repository, planner = _Repository(), _Planner(case.id)

    async def executor(step):
        return (_observation(case.session_id),)

    limits = BudgetLimits(1, 8, 0, 0, 2, 0, 120)
    outcome = await _run(_kernel(repository, planner, executor), case, limits=limits)

    assert planner.plan_calls == 1
    assert planner.compose_calls == 0, "compose ran with no model turn left"
    assert outcome.usage.model_turns == 1
    assert outcome.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert outcome.answer_candidate is None


async def test_no_model_turn_at_all_refuses_to_plan():
    case = _case()
    repository, planner = _Repository(), _Planner(case.id)

    async def executor(step):
        return ()

    with pytest.raises(BudgetExhausted):
        await _run(
            _kernel(repository, planner, executor), case,
            limits=BudgetLimits(0, 8, 0, 0, 2, 0, 120),
        )
    assert planner.plan_calls == 0, "the plan call happened before the budget check"


async def test_a_step_that_produced_nothing_does_not_answer_its_question():
    case = _case()
    repository, planner = _Repository(), _Planner(case.id)

    async def executor(step):
        return ()

    outcome = await _run(_kernel(repository, planner, executor), case)

    assert outcome.case.coverage.answered_questions == ()
    assert not outcome.case.coverage.sufficient
    assert outcome.stop_reason is not StopReason.COVERAGE_COMPLETE
    step = next(iter(repository.steps.values()))
    assert step.status is StepStatus.FAILED
    assert step.failure_reason == "success_condition_unmet"


async def test_a_step_producing_the_wrong_kind_of_observation_does_not_count():
    """`inspect_prediction` declares a prediction observation. Evidence is not
    an answer to the question it was planned for."""
    case = _case()
    repository, planner = _Repository(), _Planner(case.id)

    async def executor(step):
        return (
            Observation.create(
                session_id=case.session_id, run_id=new_id(RUN), producer=Producer.RESEARCH,
                kind=ObservationKind.EVIDENCE_SEARCH, schema_version="evidence-v1",
                canonical_payload={"hits": []}, model_projection={"hits": 0},
                provenance={"provider": "europepmc"}, now=NOW,
            ),
        )

    outcome = await _run(_kernel(repository, planner, executor), case)

    assert outcome.case.coverage.answered_questions == ()


async def test_a_satisfied_step_does_answer_its_question():
    case = _case()
    repository, planner = _Repository(), _Planner(case.id)

    async def executor(step):
        return (_observation(case.session_id),)

    outcome = await _run(_kernel(repository, planner, executor), case)

    assert outcome.case.coverage.answered_questions == ("question 0",)
    assert outcome.case.coverage.sufficient
    assert outcome.stop_reason is StopReason.COVERAGE_COMPLETE
    assert planner.compose_calls == 1


async def test_a_terminal_stop_reason_is_not_overwritten_after_the_loop():
    """Wall time runs out mid-plan. The investigation did not complete its
    coverage; reporting that it did would be a claim about the evidence."""
    case = _case(questions=("question 0", "question 1"))
    repository, planner = _Repository(), _Planner(case.id, steps=2)
    clock = {"t": 0.0}

    async def executor(step):
        clock["t"] = 999.0
        return (_observation(case.session_id),)

    outcome = await _run(
        _kernel(repository, planner, executor), case, elapsed=lambda: clock["t"]
    )

    assert outcome.stop_reason is StopReason.WALL_TIME_EXHAUSTED


async def test_the_kernel_does_not_write_completed_before_the_validators_run():
    case = _case()
    repository, planner = _Repository(), _Planner(case.id)

    async def executor(step):
        return (_observation(case.session_id),)

    kernel = _kernel(repository, planner, executor)
    outcome = await _run(kernel, case)

    assert repository.transitions[-1] is KernelState.VALIDATING_ANSWER
    assert KernelState.COMPLETED not in repository.transitions

    # A rejected candidate is a failed investigation, not a completed one.
    await kernel.record_admission(
        outcome.case, admitted=False, stop_reason=outcome.stop_reason, detail="claim_unsupported"
    )
    assert repository.transitions[-1] is KernelState.FAILED


async def test_a_restart_resumes_the_plan_and_reuses_what_it_already_produced():
    """The control plane died between two steps. The committed step's
    observations are product-owned; paying for them twice is the failure."""
    case = _case(questions=("question 0", "question 1"))
    repository, planner = _Repository(), _Planner(case.id, steps=2)
    executed: list[str] = []

    async def executor(step):
        executed.append(step.question)
        return (_observation(case.session_id),)

    kernel = _kernel(repository, planner, executor)
    plan = await planner.plan(case, ())
    await repository.save_plan(plan)
    first, second = plan.steps
    produced = _observation(case.session_id)
    repository.observations[produced.id] = produced
    await repository.save_step(plan.id, InvestigationStep(
        first.id, first.question, first.capability, first.input_refs, first.expected_output,
        first.success_condition, first.case_revision, StepStatus.COMPLETED, (produced.id,), None,
    ))
    resumed_case = case.revise(reason="plan_committed", now=NOW, plan_id=plan.id).revise(
        reason=f"step_completed:{first.id}", now=NOW,
        coverage=Coverage(case.coverage.required_questions, (first.question,)),
    )
    planner.plan_calls = 0

    outcome = await kernel.run(
        resumed_case, supported_endpoints=frozenset({"herg", "tox21"}),
        limits=LIMITS, elapsed=lambda: 0.0,
    )

    assert planner.plan_calls == 0, "a resumed case must not be replanned"
    assert executed == [second.question], "the completed step was executed again"
    assert produced.id in {item.id for item in outcome.observations}
    assert outcome.case.coverage.sufficient
    # The turn the original plan cost is still spent: a restart must not be a
    # way to buy more budget.
    assert outcome.usage.model_turns == 2
