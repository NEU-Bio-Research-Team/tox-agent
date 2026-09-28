"""One statement of what a run may spend (EffectiveRunBudgetV1).

A run's limits used to be scattered: the tool-call ceiling in
``PolicySettings``, the decision-support search/read ceilings as constants in
``tools/definitions/evidence.py``, step caps in ``RuntimeSettings``, deadlines
in two places, the identical-call loop guard in ``tools/runner.py``, and a
dormant set of numbers in ``superseded/budget.py`` that no live path reads.

Nothing here changes what is *enforced* — every limit is still enforced where
it was. This module reads those same sources and states them once, per intent,
so a run manifest, an eval manifest and the decision-support state can all
record the budget the run actually had rather than a number restated by hand.
If an enforcement site changes its constant, the snapshot changes with it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...platform.config import PolicySettings, RuntimeSettings
from ...domain.run import Intent

#: Identical arguments to the same tool this many times in one run is a loop,
#: not a retry (plan section 14.5). Enforced by ``tools/runner.py``.
MAX_IDENTICAL_CALLS = 2

#: Per-run caps on a ``decision_support`` run's evidence tools, enforced by
#: ``tools/definitions/evidence.py``: a bound on the unbounded search loop the
#: motivating run's postmortem worried about, not a limit every profile shares.
#: A starting point to tune against eval (plan section 9.2), not a permanent
#: constant. Defined here so the published budget and the enforcing tool read
#: one number.
DECISION_SUPPORT_MAX_SEARCHES_PER_RUN = 4
DECISION_SUPPORT_MAX_EVIDENCE_READS_PER_RUN = 8

SCHEMA_VERSION = "effective-run-budget-v1"


@dataclass(frozen=True, slots=True)
class EffectiveRunBudgetV1:
    intent: str
    #: Wall clock for the whole run, in seconds.
    deadline_s: int
    #: Runtime turn deadline (one model turn), in seconds; ``None`` for a lane
    #: that never dispatches a model.
    turn_deadline_s: int | None
    #: What the control plane asks the runtime for. The effective cap the
    #: deployed profile enforces is recorded separately by the gateway.
    requested_step_cap: int | None
    #: Read/search tool calls; the submit tools are exempt (tools/runner.py).
    max_tool_calls: int | None
    #: ``None`` means the lane does not bound searches separately.
    max_searches: int | None
    max_evidence_reads: int | None
    #: Identical arguments to the same tool more than this is a loop.
    max_identical_calls: int | None
    #: Answer candidates, including the one correction attempt.
    max_answer_candidates: int | None
    #: Correction turns a refused submission may spend.
    max_correction_turns: int | None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DETERMINISTIC = {
    Intent.ANALYSIS.value, Intent.ANALYSIS_BATCH.value,
    Intent.STRUCTURE_RECOGNITION.value,
    Intent.CLARIFICATION_REQUIRED.value, Intent.OUT_OF_SCOPE.value,
}


def effective_run_budget(
    intent: str, policy: PolicySettings, runtime: RuntimeSettings
) -> EffectiveRunBudgetV1:
    """The budget a run of ``intent`` gets under these settings."""
    if intent in _DETERMINISTIC:
        deadline = (
            policy.structure_recognition_deadline_s
            if intent == Intent.STRUCTURE_RECOGNITION.value
            else policy.run_deadline_s
        )
        return EffectiveRunBudgetV1(
            intent=intent, deadline_s=deadline, turn_deadline_s=None,
            requested_step_cap=None, max_tool_calls=None, max_searches=None,
            max_evidence_reads=None, max_identical_calls=None,
            max_answer_candidates=None, max_correction_turns=None,
        )
    if intent == Intent.BUILD_REPORT.value:
        return EffectiveRunBudgetV1(
            intent=intent,
            deadline_s=policy.report_build_deadline_s,
            turn_deadline_s=runtime.report_turn_deadline_s,
            requested_step_cap=runtime.max_steps_report,
            max_tool_calls=policy.max_tool_calls_per_report_build,
            max_searches=None, max_evidence_reads=None,
            max_identical_calls=MAX_IDENTICAL_CALLS,
            max_answer_candidates=None,
            # report_synthesis.py: one refused submission may be corrected.
            max_correction_turns=1,
        )
    decision_support = intent == Intent.DECISION_SUPPORT.value
    return EffectiveRunBudgetV1(
        intent=intent,
        deadline_s=policy.run_deadline_s,
        turn_deadline_s=runtime.turn_deadline_s,
        requested_step_cap=(
            runtime.max_steps_research
            if intent in (Intent.EVIDENCE_RESEARCH.value, Intent.DECISION_SUPPORT.value)
            else runtime.max_steps_qa
        ),
        max_tool_calls=policy.max_tool_calls_per_run,
        max_searches=DECISION_SUPPORT_MAX_SEARCHES_PER_RUN if decision_support else None,
        max_evidence_reads=(
            DECISION_SUPPORT_MAX_EVIDENCE_READS_PER_RUN if decision_support else None
        ),
        max_identical_calls=MAX_IDENTICAL_CALLS,
        max_answer_candidates=policy.max_answer_candidates_per_run,
        max_correction_turns=max(0, policy.max_answer_candidates_per_run - 1),
    )


def budget_matrix(policy: PolicySettings, runtime: RuntimeSettings) -> dict[str, dict[str, Any]]:
    """Every live intent's budget, for a manifest."""
    return {
        intent.value: effective_run_budget(intent.value, policy, runtime).to_dict()
        for intent in Intent
        if intent not in (Intent.REPORT_QA, Intent.EVIDENCE_RESEARCH, Intent.ATTRIBUTION)
    }
