"""Global, per-tenant, per-provider and per-queue caps (WS08 / PR-15).

A cap is only a cap if every worker counts against the same number. So the
count lives in ``concurrency_slots`` and a worker takes a slot the way it takes
a job: by a write whose condition is the mutual exclusion. There is no
"read how many are running, then decide" step anywhere, because that step is
exactly where two workers both conclude there is room.

Four scopes, all optional:

* ``global`` — every run, whatever it is. Protects the database and the host.
* ``queue`` — one queue class. The report cap is what a model provider's rate
  limit actually tolerates for long builds.
* ``tenant`` — one owner. One user's batch of fifty cannot occupy the fleet.
* ``provider`` — one AI profile. Deterministic runs do not count here: they
  never reach a model.

A worker that cannot take every slot a run needs takes none of them and puts
the job back with a short delay. Holding some slots while waiting on others is
how two runs deadlock each other.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Mapping

from .queues import QueueClass

log = logging.getLogger("toxagent.concurrency")


@dataclass(frozen=True, slots=True)
class SlotRequest:
    scope: str
    key: str
    limit: int


@dataclass(frozen=True)
class ConcurrencyLimits:
    """Zero or absent means no cap in that scope."""

    global_max: int = 0
    per_tenant: int = 0
    per_provider: int = 0
    per_queue: Mapping[str, int] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return bool(
            self.global_max or self.per_tenant or self.per_provider
            or any(v > 0 for v in self.per_queue.values())
        )

    def requests_for(
        self, *, tenant: str, provider: str | None, queue_name: str
    ) -> tuple[SlotRequest, ...]:
        """Slots a run needs, in a fixed order.

        The order is the same for every run, which is what makes "take all or
        none" free of lock-order inversions: nobody holds tenant while waiting
        on global.
        """
        requests: list[SlotRequest] = []
        if self.global_max > 0:
            requests.append(SlotRequest("global", "*", self.global_max))
        queue_cap = self.per_queue.get(queue_name, 0)
        if queue_cap > 0:
            requests.append(SlotRequest("queue", queue_name, queue_cap))
        if self.per_provider > 0 and provider and queue_name != QueueClass.DETERMINISTIC.value:
            requests.append(SlotRequest("provider", provider[:128], self.per_provider))
        if self.per_tenant > 0:
            requests.append(SlotRequest("tenant", tenant[:128], self.per_tenant))
        return tuple(requests)


@dataclass(frozen=True, slots=True)
class SlotDecision:
    granted: bool
    #: The scope that refused, when one did. Low-cardinality by construction.
    refused_scope: str | None = None


class SlotLeaser:
    def __init__(self, database, limits: ConcurrencyLimits, *, ttl_s: float) -> None:
        self._db = database
        self._limits = limits
        self._ttl = timedelta(seconds=ttl_s)

    @property
    def limits(self) -> ConcurrencyLimits:
        return self._limits

    async def acquire(
        self,
        *,
        run_id: str,
        worker_id: str,
        tenant: str,
        provider: str | None,
        queue_name: str,
        now: datetime,
    ) -> SlotDecision:
        requests = self._limits.requests_for(
            tenant=tenant, provider=provider, queue_name=queue_name
        )
        if not requests:
            return SlotDecision(True)
        async with self._db.unit_of_work() as uow:
            for request in requests:
                taken = await uow.concurrency_slots.take(
                    scope=request.scope,
                    scope_key=request.key,
                    limit=request.limit,
                    run_id=run_id,
                    worker_id=worker_id,
                    expires_at=now + self._ttl,
                    now=now,
                )
                if not taken:
                    # Leaving the unit of work uncommitted rolls back every
                    # slot taken above: all or none.
                    return SlotDecision(False, request.scope)
            await uow.commit()
        return SlotDecision(True)

    async def renew(self, *, run_id: str, worker_id: str, now: datetime) -> None:
        async with self._db.unit_of_work() as uow:
            await uow.concurrency_slots.renew(
                run_id=run_id, worker_id=worker_id, expires_at=now + self._ttl
            )
            await uow.commit()

    async def release(self, *, run_id: str, worker_id: str) -> None:
        async with self._db.unit_of_work() as uow:
            await uow.concurrency_slots.release(run_id=run_id, worker_id=worker_id)
            await uow.commit()


def limits_from_settings(worker_settings) -> ConcurrencyLimits:
    return ConcurrencyLimits(
        global_max=worker_settings.global_max_runs,
        per_tenant=worker_settings.tenant_max_runs,
        per_provider=worker_settings.provider_max_runs,
        per_queue={QueueClass.REPORT.value: worker_settings.report_max_runs},
    )
