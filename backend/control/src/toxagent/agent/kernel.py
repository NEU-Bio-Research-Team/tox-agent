"""Runtime-neutral orchestration for one scientific investigation.

The kernel accepts structured plans and semantic drafts from replaceable model
components, while every transition, capability execution, budget check and
answer compilation remains product-owned and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Awaitable, Callable, Protocol, Sequence

from ..answer.compiler import AnswerCompiler, SemanticAnswerDraft
from ..capabilities.registry import CapabilityRegistry
from ..domain.investigation import CaseState, Coverage, InvestigationPlan, InvestigationStep, StepStatus
from ..domain.observation import Observation
from .budget import BudgetLimits, BudgetUsage, StopReason, can_consume, stop_reason


class KernelState(str, Enum):
    LOADING = "loading"
    PLANNING = "planning"
    VALIDATING_PLAN = "validating_plan"
    EXECUTING = "executing"
    CHECKING_COVERAGE = "checking_coverage"
    RESOLVING_CONFLICT = "resolving_conflict"
    COMPOSING = "composing"
    VALIDATING_ANSWER = "validating_answer"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class KernelTransition:
    state: KernelState
    occurred_at: datetime
    detail: str = ""


class CaseRepository(Protocol):
    async def save_case(self, case: CaseState, *, expected_revision: int | None) -> None: ...
    async def save_plan(self, plan: InvestigationPlan) -> None: ...
    async def save_step(self, plan_id: str, step: InvestigationStep) -> None: ...
    async def append_transition(self, case_id: str, transition: KernelTransition) -> None: ...


class Planner(Protocol):
    async def plan(self, case: CaseState, capabilities: Sequence[dict[str, object]]) -> InvestigationPlan: ...
    async def compose(self, case: CaseState, observations: Sequence[Observation]) -> SemanticAnswerDraft: ...


CapabilityExecutor = Callable[[InvestigationStep], Awaitable[tuple[Observation, ...]]]


@dataclass(frozen=True, slots=True)
class KernelOutcome:
    case: CaseState
    plan: InvestigationPlan
    observations: tuple[Observation, ...]
    answer_candidate: object
    stop_reason: StopReason
    usage: BudgetUsage


class ScientificAgentKernel:
    def __init__(self, *, capabilities: CapabilityRegistry, repository: CaseRepository,
                 planner: Planner, execute_capability: CapabilityExecutor,
                 compiler: AnswerCompiler, clock: Callable[[], datetime]) -> None:
        self._capabilities = capabilities
        self._repository = repository
        self._planner = planner
        self._execute = execute_capability
        self._compiler = compiler
        self._clock = clock

    async def _transition(self, case_id: str, state: KernelState, detail: str = "") -> None:
        await self._repository.append_transition(case_id, KernelTransition(state, self._clock(), detail))

    async def run(self, case: CaseState, *, supported_endpoints: frozenset[str],
                  limits: BudgetLimits, elapsed: Callable[[], float]) -> KernelOutcome:
        await self._transition(case.id, KernelState.LOADING)
        await self._transition(case.id, KernelState.PLANNING)
        plan = await self._planner.plan(case, list(self._capabilities.planner_view()))
        usage = BudgetUsage(model_turns=1)
        await self._transition(case.id, KernelState.VALIDATING_PLAN)
        self._capabilities.validate_plan(plan, goal=case.goal, supported_endpoints=supported_endpoints,
                                         max_replans=limits.replans)
        await self._repository.save_plan(plan)
        previous_revision = case.revision
        case = case.revise(reason="plan_committed", now=self._clock(), plan_id=plan.id)
        await self._repository.save_case(case, expected_revision=previous_revision)

        observations: list[Observation] = []
        reason = StopReason.CONTINUE
        for step in plan.steps:
            reason = stop_reason(limits, usage, coverage_sufficient=case.coverage.sufficient,
                                 elapsed_s=elapsed())
            if reason is not StopReason.CONTINUE:
                break
            definition = self._capabilities.get(step.capability)
            tools = definition.tool_sequence
            costs = {
                "tool_calls": len(tools),
                "searches": sum(name.startswith("search_") for name in tools),
                "reads": sum(name == "get_evidence_record" for name in tools),
                "attributions": sum(name in {"get_attribution", "get_explanation_slice"} for name in tools),
            }
            if not can_consume(limits, usage, **costs):
                reason = StopReason.BUDGET_EXHAUSTED
                break
            running = InvestigationStep(
                step.id, step.question, step.capability, step.input_refs, step.expected_output,
                step.success_condition, step.case_revision, StepStatus.RUNNING,
            )
            await self._repository.save_step(plan.id, running)
            await self._transition(case.id, KernelState.EXECUTING, step.id)
            try:
                produced = await self._execute(running)
            except Exception as exc:
                failed = InvestigationStep(
                    step.id, step.question, step.capability, step.input_refs, step.expected_output,
                    step.success_condition, step.case_revision, StepStatus.FAILED, (), type(exc).__name__,
                )
                await self._repository.save_step(plan.id, failed)
                raise
            for field, amount in costs.items():
                usage = usage.consume(field, amount)
            observations.extend(produced)
            completed = InvestigationStep(
                step.id, step.question, step.capability, step.input_refs, step.expected_output,
                step.success_condition, step.case_revision, StepStatus.COMPLETED,
                tuple(observation.id for observation in produced), None,
            )
            await self._repository.save_step(plan.id, completed)
            answered = tuple(dict.fromkeys((*case.coverage.answered_questions, step.question)))
            previous_revision = case.revision
            case = case.revise(
                reason=f"step_completed:{step.id}", now=self._clock(),
                coverage=Coverage(
                    case.coverage.required_questions, answered,
                    case.coverage.unresolved_conflict_ids, case.coverage.blocking_gap_ids,
                ),
            )
            await self._repository.save_case(case, expected_revision=previous_revision)
            await self._transition(case.id, KernelState.CHECKING_COVERAGE, step.id)

        reason = stop_reason(limits, usage, coverage_sufficient=case.coverage.sufficient,
                             elapsed_s=elapsed())
        await self._transition(case.id, KernelState.COMPOSING, reason.value)
        draft = await self._planner.compose(case, observations)
        usage = usage.consume("model_turns")
        candidate = self._compiler.compile(draft, observations={item.id: item for item in observations})
        await self._transition(case.id, KernelState.VALIDATING_ANSWER)
        # Existing deterministic answer validators own the actual admission;
        # the kernel returns their canonical wire candidate, never commits text.
        await self._transition(case.id, KernelState.COMPLETED, reason.value)
        return KernelOutcome(case, plan, tuple(observations), candidate, reason, usage)
