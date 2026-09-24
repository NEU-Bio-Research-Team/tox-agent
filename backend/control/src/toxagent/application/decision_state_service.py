"""Where DecisionSupportStateV1 transitions meet the database.

The domain module owns the transitions; this module owns reading, applying and
writing them with a revision check, and the rule that state bookkeeping never
breaks the run it observes. A tool call, an answer or a run end that could not
update the state still happens — the state records what it could, and a
failure to record is logged, not raised into the product path.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

from ..domain import decision_state as ds
from ..domain.errors import Conflict

log = logging.getLogger("toxagent.decision_state")

Transition = Callable[[ds.DecisionSupportStateV1], ds.DecisionSupportStateV1]


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def begin(
    uow, *, session_id: str, run_id: str, goal: str, subject_refs: Iterable[str],
    budget_snapshot: Mapping[str, Any],
) -> ds.DecisionSupportStateV1:
    """The run's initial state. Idempotent: a re-dispatched run keeps its own."""
    existing = await uow.decision_states.get(run_id)
    if existing is not None:
        return existing
    state = ds.initial(
        session_id=session_id, run_id=run_id, goal=goal, subject_refs=subject_refs,
        budget_snapshot=budget_snapshot,
    )
    await uow.decision_states.put(state, expected_revision=None, now=_now())
    return state


async def advance_in(uow, run_id: str, transition: Transition) -> ds.DecisionSupportStateV1 | None:
    """Apply one transition inside the caller's unit of work.

    ``None`` when the run has no state (not a decision_support run). Raises
    ``Conflict`` on a concurrent write and ``InvalidPlan`` on a refused
    transition; the caller decides whether either is fatal.
    """
    state = await uow.decision_states.get(run_id)
    if state is None:
        return None
    updated = transition(state)
    if updated is state:
        return state
    await uow.decision_states.put(updated, expected_revision=state.revision, now=_now())
    return updated


async def advance(database, run_id: str, transition: Transition, *, attempts: int = 4) -> None:
    """Apply one transition in its own unit of work, retrying a lost race.

    Used where the state is observational: a failure is logged and swallowed.
    """
    for attempt in range(attempts):
        try:
            async with database.unit_of_work() as uow:
                await advance_in(uow, run_id, transition)
                await uow.commit()
            return
        except Conflict:
            if attempt == attempts - 1:
                log.warning("decision state update lost %d races; skipped", attempts,
                            extra={"run_id": run_id})
        except Exception:  # noqa: BLE001 - bookkeeping must not fail the run
            log.exception("decision state update failed", extra={"run_id": run_id})
            return


def relations_payload(relations: Iterable[Any]) -> list[dict[str, Any]]:
    """Wire relations (EvidenceRelationInputV2) as the transition reads them."""
    return [
        {"proposition": r.proposition, "source_class": r.source_class,
         "source_id": r.source_id, "relation": r.relation}
        for r in relations
    ]
