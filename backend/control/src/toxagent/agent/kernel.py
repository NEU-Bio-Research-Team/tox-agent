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
from .budget import (
    BudgetExhausted, BudgetLimits, BudgetUsage, StopReason, can_consume, stop_reason,
)


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
    async def get_plan(self, plan_id: str) -> InvestigationPlan | None: ...
    async def load_observations(self, ids: Sequence[str]) -> tuple[Observation, ...]: ...


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
        # I31: a control plane that restarted mid-investigation could only
        # plan again — a fresh model turn, and every completed step's
        # predictor call or evidence retrieval paid for a second time, with a
        # different plan to run them under. A case that already committed one
        # resumes it: the plan and its steps are durable, and the observations
        # a completed step produced are immutable and product-owned, so they
        # are read rather than recomputed.
        resumed = await self._repository.get_plan(case.plan_id) if case.plan_id else None
        if resumed is not None:
            # Re-validated against *this* deployment: the plan was admissible
            # when it was written, and an endpoint or capability may have gone
            # away since. A resumed plan is not exempt from the check a fresh
            # one has to pass.
            await self._transition(case.id, KernelState.VALIDATING_PLAN, "resumed")
            self._capabilities.validate_plan(
                resumed, goal=case.goal, supported_endpoints=supported_endpoints,
                max_replans=limits.replans,
            )
            return await self._continue(case, resumed, limits=limits, elapsed=elapsed)
        await self._transition(case.id, KernelState.PLANNING)
        usage = BudgetUsage()
        # I31: the plan call used to happen first and the turn was recorded
        # afterwards, so a limit of one model turn produced two calls — the
        # plan, then compose. Reserving before the call is what makes the
        # limit a limit: the usage says a turn is spent whether or not the
        # call comes back.
        if not can_consume(limits, usage, model_turns=1):
            raise BudgetExhausted(
                "the model-turn budget does not allow planning this investigation",
                limit=limits.model_turns,
            )
        usage = usage.consume("model_turns")
        plan = await self._planner.plan(case, list(self._capabilities.planner_view()))
        await self._transition(case.id, KernelState.VALIDATING_PLAN)
        self._capabilities.validate_plan(plan, goal=case.goal, supported_endpoints=supported_endpoints,
                                         max_replans=limits.replans)
        await self._repository.save_plan(plan)
        previous_revision = case.revision
        case = case.revise(reason="plan_committed", now=self._clock(), plan_id=plan.id)
        await self._repository.save_case(case, expected_revision=previous_revision)
        return await self._continue(case, plan, limits=limits, elapsed=elapsed, usage=usage)

    @staticmethod
    def _step_costs(tools: Sequence[str]) -> dict[str, int]:
        return {
            "tool_calls": len(tools),
            "searches": sum(name.startswith("search_") for name in tools),
            "reads": sum(name == "get_evidence_record" for name in tools),
            "attributions": sum(
                name in {"get_attribution", "get_explanation_slice"} for name in tools
            ),
        }

    def _usage_already_spent(self, plan: InvestigationPlan) -> BudgetUsage:
        """What a resumed investigation has already cost.

        The plan that exists cost a model turn, and every step that ran cost
        its capability's tools. Starting a resumed run from zero would let a
        restart loop reset the budget — the crash would become a way to buy
        more of it.
        """
        usage = BudgetUsage(model_turns=1)
        for step in plan.steps:
            if step.status in (StepStatus.PENDING, StepStatus.RUNNING):
                continue
            costs = self._step_costs(self._capabilities.get(step.capability).tool_sequence)
            for field, amount in costs.items():
                usage = usage.consume(field, amount)
        return usage

    async def _continue(
        self, case: CaseState, plan: InvestigationPlan, *, limits: BudgetLimits,
        elapsed: Callable[[], float], usage: BudgetUsage | None = None,
    ) -> KernelOutcome:
        """Execute the plan's outstanding steps and compose.

        Shared by a fresh run and a resumed one; the only difference is where
        the plan and the already-spent budget came from.
        """
        if usage is None:
            usage = self._usage_already_spent(plan)
        observations: list[Observation] = []
        # The reason the loop stopped, once it has one. Recomputing it after
        # the loop is what let a terminal `budget_exhausted` be overwritten by
        # a later `continue` or `coverage_complete` (I31) — the kernel then
        # reported the investigation as having finished on its own terms.
        reason = StopReason.CONTINUE
        for step in plan.steps:
            if step.status is StepStatus.COMPLETED:
                # Already done, and its observations are immutable: read them
                # back instead of paying the predictor or the evidence
                # provider again for an answer this case already holds.
                observations.extend(await self._repository.load_observations(step.output_refs))
                continue
            if step.status is StepStatus.FAILED:
                # It ran and did not satisfy its capability. Re-running it
                # here would be a retry loop nothing asked for; replanning is
                # the deliberate path, under `limits.replans`.
                continue
            reason = stop_reason(limits, usage, coverage_sufficient=case.coverage.sufficient,
                                 elapsed_s=elapsed())
            if reason is not StopReason.CONTINUE:
                break
            definition = self._capabilities.get(step.capability)
            costs = self._step_costs(definition.tool_sequence)
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
            # I31: a step that returned nothing used to be marked completed
            # and its question marked answered, so coverage read as sufficient
            # on no evidence at all. `satisfied_by` is a floor — at least one
            # observation of a kind this capability declares it produces — but
            # it is a check, which is what the free-text success condition
            # never was.
            satisfied = definition.satisfied_by(produced)
            await self._repository.save_step(plan.id, InvestigationStep(
                step.id, step.question, step.capability, step.input_refs, step.expected_output,
                step.success_condition, step.case_revision,
                StepStatus.COMPLETED if satisfied else StepStatus.FAILED,
                tuple(observation.id for observation in produced),
                None if satisfied else "success_condition_unmet",
            ))
            if not satisfied:
                # The step ran and cost its budget; the question it was for is
                # still open, and saying so is what keeps a later
                # `coverage_complete` honest.
                await self._transition(case.id, KernelState.CHECKING_COVERAGE, step.id)
                continue
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

        if reason is StopReason.CONTINUE:
            # The loop ran out of steps rather than out of anything else.
            reason = stop_reason(limits, usage, coverage_sufficient=case.coverage.sufficient,
                                 elapsed_s=elapsed())

        # Composing is a model call like planning was, and needs a turn
        # reserved the same way (I31). Without one there is no answer to
        # return, and returning a candidate anyway would mean the budget only
        # ever applied to work nobody was watching.
        if not can_consume(limits, usage, model_turns=1):
            await self._transition(
                case.id, KernelState.COMPOSING, StopReason.BUDGET_EXHAUSTED.value
            )
            return KernelOutcome(
                case, plan, tuple(observations), None, StopReason.BUDGET_EXHAUSTED, usage
            )
        usage = usage.consume("model_turns")
        await self._transition(case.id, KernelState.COMPOSING, reason.value)
        draft = await self._planner.compose(case, observations)
        candidate = self._compiler.compile(draft, observations={item.id: item for item in observations})
        # The kernel stops here. Admission belongs to the deterministic answer
        # validators, and the kernel used to write COMPLETED before they had
        # run — a terminal state asserting an outcome nothing had decided.
        # `record_admission` writes that transition, from whoever knows.
        await self._transition(case.id, KernelState.VALIDATING_ANSWER)
        return KernelOutcome(case, plan, tuple(observations), candidate, reason, usage)

    async def record_admission(
        self, case: CaseState, *, admitted: bool, stop_reason: StopReason, detail: str = ""
    ) -> None:
        """Write the terminal transition, once the validators have decided.

        Split out of `run` because the kernel does not validate answers and
        must not claim one was accepted (I31). A rejected candidate is a
        failed investigation, not a completed one with an unused answer.
        """
        await self._transition(
            case.id,
            KernelState.COMPLETED if admitted else KernelState.FAILED,
            detail or stop_reason.value,
        )
