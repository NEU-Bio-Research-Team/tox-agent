"""In-process wake-ups for the change feed.

The outbox is the source of truth; this is only latency. A subscriber that
misses a notification still catches up on its next poll, and a subscriber that
receives a spurious one finds nothing new — so nothing in the product's
correctness depends on this class working.

It is process-local by design. Several API instances each read the same outbox;
adding a cross-process bus (LISTEN/NOTIFY, Redis) is a latency optimisation to
make when telemetry says the poll interval is costing something, and the
interface here does not change when it happens.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Sequence


class EventNotifier:
    def __init__(self) -> None:
        self._waiters: dict[str, set[asyncio.Event]] = defaultdict(set)

    def notify(self, session_ids: Sequence[str]) -> None:
        for session_id in session_ids:
            for waiter in self._waiters.get(session_id, ()):
                waiter.set()

    async def wait(self, session_id: str, timeout: float) -> bool:
        """Block until something commits for this session, or the timeout.

        Returns whether a notification arrived, which the caller uses only to
        decide how eagerly to poll — never to decide whether to poll.
        """
        waiter = asyncio.Event()
        self._waiters[session_id].add(waiter)
        try:
            await asyncio.wait_for(waiter.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            self._waiters[session_id].discard(waiter)
            if not self._waiters[session_id]:
                self._waiters.pop(session_id, None)

    @property
    def subscriber_count(self) -> int:
        return sum(len(w) for w in self._waiters.values())
