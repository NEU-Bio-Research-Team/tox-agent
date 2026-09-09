"""In-flight cap for the stateless ``/v1/predict*`` routes (plan section 3.2).

Quick Predict deliberately bypasses the session lifecycle, so it also bypasses
the concurrent-run admission guard that protects ToxPred on the session path.
This is the replacement: a per-principal semaphore acquired without waiting, so a
caller that floods the predictor gets ``429 provider_rate_limited`` immediately
rather than queueing work onto the model.

Process-local by construction. A deployment running ``N`` control-plane
instances behind a load balancer therefore allows up to
``N x max_inflight_per_principal`` concurrent calls for one principal; a global
limiter is folded into the W9 abuse-control work and is out of scope here.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from ..domain.errors import InvalidRequest, ProviderRateLimited


class PredictBatchTooLarge(InvalidRequest):
    """A well-formed batch request whose length the deployment will not serve.

    422 rather than 400: the body parses, the endpoints are valid, only the size
    is out of range.
    """

    http_status = 422


class PredictLimiter:
    def __init__(self, *, max_inflight_per_principal: int, max_batch_size: int) -> None:
        if max_inflight_per_principal < 1:
            raise ValueError("max_inflight_per_principal must be at least 1")
        self._max_inflight = max_inflight_per_principal
        self._max_batch = max_batch_size
        #: One semaphore per principal, created on first use. The map only ever
        #: grows; each entry is a couple of pointers, and a principal set large
        #: enough to matter is itself an abuse signal the W9 work will address.
        self._slots: dict[str, asyncio.Semaphore] = {}

    @property
    def max_batch_size(self) -> int:
        return self._max_batch

    def check_batch_size(self, count: int) -> None:
        """A batch larger than the predictor's documented maximum is refused
        before any forward pass. ``count`` of zero is a malformed request."""
        if count < 1:
            raise InvalidRequest("a batch needs at least one SMILES")
        if count > self._max_batch:
            raise PredictBatchTooLarge(
                f"a batch of {count} exceeds the maximum of {self._max_batch}",
                max_batch_size=self._max_batch,
                count=count,
            )

    @asynccontextmanager
    async def slot(self, principal_id: str) -> AsyncIterator[None]:
        """Hold one in-flight slot for ``principal_id`` for the duration of the
        block, or raise ``ProviderRateLimited`` if the principal is already at
        the cap. Non-blocking: this never queues a request behind another."""
        sem = self._slots.setdefault(principal_id, asyncio.Semaphore(self._max_inflight))
        # Single-threaded asyncio: nothing else runs between ``locked()`` and the
        # non-suspending fast path of ``acquire()`` when a slot is free, so this
        # check-then-acquire is not a race.
        if sem.locked():
            raise ProviderRateLimited(
                "too many predict calls in flight for this principal",
                max_inflight=self._max_inflight,
            )
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()
