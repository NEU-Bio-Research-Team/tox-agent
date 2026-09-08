"""Short-transaction durable repository used by the scientific agent kernel."""
from __future__ import annotations

from ..agent.kernel import KernelTransition
from ..domain.investigation import CaseState, InvestigationPlan, InvestigationStep


class DurableInvestigationRepository:
    def __init__(self, database) -> None:
        self._database = database

    async def save_case(self, case: CaseState, *, expected_revision: int | None) -> None:
        async with self._database.unit_of_work() as uow:
            await uow.investigations.save_case(case, expected_revision=expected_revision)
            await uow.commit()

    async def save_plan(self, plan: InvestigationPlan) -> None:
        async with self._database.unit_of_work() as uow:
            await uow.investigations.save_plan(plan)
            await uow.commit()

    async def save_step(self, plan_id: str, step: InvestigationStep) -> None:
        async with self._database.unit_of_work() as uow:
            await uow.investigations.save_step(plan_id, step)
            await uow.commit()

    async def append_transition(self, case_id: str, transition: KernelTransition) -> None:
        async with self._database.unit_of_work() as uow:
            await uow.investigations.append_transition(case_id, transition)
            await uow.commit()
