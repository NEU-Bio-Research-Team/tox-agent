"""Deterministic investigation budgets and stop decisions."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    model_turns: int
    tool_calls: int
    searches: int
    reads: int
    attributions: int
    replans: int
    wall_time_s: int


PROFILES = {
    "report_qa": BudgetLimits(4, 8, 0, 0, 1, 0, 120),
    "evidence_research": BudgetLimits(10, 24, 4, 12, 2, 2, 600),
    "attribution": BudgetLimits(4, 8, 0, 0, 1, 0, 180),
}


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    model_turns: int = 0
    tool_calls: int = 0
    searches: int = 0
    reads: int = 0
    attributions: int = 0
    replans: int = 0

    def consume(self, kind: str, amount: int = 1) -> "BudgetUsage":
        if kind not in self.__dataclass_fields__ or amount < 0:
            raise ValueError(f"invalid budget consumption: {kind}={amount}")
        return replace(self, **{kind: getattr(self, kind) + amount})


class BudgetExhausted(RuntimeError):
    """No budget remains for a call the kernel must make to proceed.

    Distinct from `StopReason.BUDGET_EXHAUSTED`, which describes an
    investigation that ran and then stopped: this one cannot start.
    """

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.context = context


class StopReason(str, Enum):
    COVERAGE_COMPLETE = "coverage_complete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    WALL_TIME_EXHAUSTED = "wall_time_exhausted"
    CONTINUE = "continue"


def stop_reason(limits: BudgetLimits, usage: BudgetUsage, *, coverage_sufficient: bool,
                elapsed_s: float) -> StopReason:
    if coverage_sufficient:
        return StopReason.COVERAGE_COMPLETE
    if elapsed_s >= limits.wall_time_s:
        return StopReason.WALL_TIME_EXHAUSTED
    for field in BudgetUsage.__dataclass_fields__:
        if getattr(usage, field) > getattr(limits, field):
            return StopReason.BUDGET_EXHAUSTED
    return StopReason.CONTINUE


def can_consume(limits: BudgetLimits, usage: BudgetUsage, **amounts: int) -> bool:
    return all(
        getattr(usage, field) + amount <= getattr(limits, field)
        for field, amount in amounts.items()
    )
