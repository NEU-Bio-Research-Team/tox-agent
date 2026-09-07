"""Bounded retry on the pre-flight runtime health probe.

progress log §3.8/§5.2: killing OpenCode mid-turn and restarting it produced a
recovery run that was itself created correctly but failed `runtime_unavailable`
because the gateway's single point-in-time health check raced the ~2s the
runtime needed to finish loading after restart. `_probe_health_with_retries`
gives that window a bounded number of chances to close without weakening the
check for a runtime that is genuinely down.
"""
from __future__ import annotations

import pytest

from toxagent.config import RuntimeSettings
from toxagent.harness.gateway import AgentRuntimeGateway
from toxagent.harness.provider import RuntimeHealth

pytestmark = pytest.mark.anyio


class _FlakyHealthProvider:
    """Unhealthy for the first ``fail_times`` probes, then healthy — standing
    in for a runtime that was just restarted and needs a moment before it can
    serve a real health check."""

    kind = "scripted"

    def __init__(self, fail_times: int, raise_instead: bool = False) -> None:
        self.fail_times = fail_times
        self.raise_instead = raise_instead
        self.calls = 0

    async def health(self) -> RuntimeHealth:
        self.calls += 1
        if self.calls <= self.fail_times:
            if self.raise_instead:
                raise RuntimeError("connection refused")
            return RuntimeHealth(healthy=False, detail="not ready yet")
        return RuntimeHealth(healthy=True, detail="ready")


def _gateway(provider, **settings_overrides) -> AgentRuntimeGateway:
    settings = RuntimeSettings(runtime_health_check_retry_delay_s=0.0, **settings_overrides)
    return AgentRuntimeGateway(
        database=None,
        registry=None,
        capability_tokens=None,
        provider=provider,
        settings=settings,
    )


async def test_a_transient_unhealthy_probe_recovers_within_the_retry_budget():
    provider = _FlakyHealthProvider(fail_times=2)
    gateway = _gateway(provider, runtime_health_check_retries=3)
    health = await gateway._probe_health_with_retries()
    assert health.healthy
    assert provider.calls == 3


async def test_a_probe_that_raises_is_retried_the_same_as_one_that_returns_unhealthy():
    provider = _FlakyHealthProvider(fail_times=1, raise_instead=True)
    gateway = _gateway(provider, runtime_health_check_retries=2)
    health = await gateway._probe_health_with_retries()
    assert health.healthy
    assert provider.calls == 2


async def test_a_runtime_still_down_after_every_attempt_stays_unhealthy():
    provider = _FlakyHealthProvider(fail_times=10)
    gateway = _gateway(provider, runtime_health_check_retries=3)
    health = await gateway._probe_health_with_retries()
    assert not health.healthy
    assert health.detail
    assert provider.calls == 3


async def test_one_retry_is_the_old_no_retry_behaviour():
    provider = _FlakyHealthProvider(fail_times=1)
    gateway = _gateway(provider, runtime_health_check_retries=1)
    health = await gateway._probe_health_with_retries()
    assert not health.healthy
    assert provider.calls == 1


async def test_a_healthy_first_probe_does_not_retry_at_all():
    provider = _FlakyHealthProvider(fail_times=0)
    gateway = _gateway(provider, runtime_health_check_retries=5)
    health = await gateway._probe_health_with_retries()
    assert health.healthy
    assert provider.calls == 1
