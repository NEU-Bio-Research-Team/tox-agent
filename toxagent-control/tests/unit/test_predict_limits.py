"""The in-flight limiter for the ``/v1/predict*`` routes (plan section 3.2)."""
from __future__ import annotations

import asyncio

import pytest

from toxagent.api.predict_limits import PredictBatchTooLarge, PredictLimiter
from toxagent.domain.errors import InvalidRequest, ProviderRateLimited

pytestmark = pytest.mark.anyio


async def test_a_call_over_the_in_flight_cap_is_rate_limited():
    limiter = PredictLimiter(max_inflight_per_principal=4, max_batch_size=64)
    release = asyncio.Event()

    async def hold():
        async with limiter.slot("user-1"):
            await release.wait()

    holders = [asyncio.create_task(hold()) for _ in range(4)]
    await asyncio.sleep(0.02)  # let all four acquire their slot

    with pytest.raises(ProviderRateLimited):
        async with limiter.slot("user-1"):
            pass

    release.set()
    await asyncio.gather(*holders)


async def test_capacity_is_released_when_the_block_exits():
    limiter = PredictLimiter(max_inflight_per_principal=1, max_batch_size=64)
    async with limiter.slot("user-1"):
        pass
    # A second acquisition would raise if the first had not been released.
    async with limiter.slot("user-1"):
        pass


async def test_each_principal_has_its_own_capacity():
    limiter = PredictLimiter(max_inflight_per_principal=1, max_batch_size=64)
    release = asyncio.Event()

    async def hold(principal):
        async with limiter.slot(principal):
            await release.wait()

    first = asyncio.create_task(hold("user-1"))
    await asyncio.sleep(0.02)
    # user-2 is unaffected by user-1 sitting at its cap.
    async with limiter.slot("user-2"):
        pass
    release.set()
    await first


def test_a_batch_larger_than_the_maximum_is_422():
    limiter = PredictLimiter(max_inflight_per_principal=4, max_batch_size=3)
    with pytest.raises(PredictBatchTooLarge) as raised:
        limiter.check_batch_size(4)
    assert raised.value.http_status == 422
    assert raised.value.code == "invalid_request"


def test_an_empty_batch_is_rejected():
    limiter = PredictLimiter(max_inflight_per_principal=4, max_batch_size=3)
    with pytest.raises(InvalidRequest):
        limiter.check_batch_size(0)


def test_a_batch_at_the_maximum_is_allowed():
    limiter = PredictLimiter(max_inflight_per_principal=4, max_batch_size=3)
    limiter.check_batch_size(3)  # no raise
