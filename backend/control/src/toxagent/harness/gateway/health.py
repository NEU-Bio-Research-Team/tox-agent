"""Runtime health, identity and limits as the gateway reports them."""
from __future__ import annotations

import asyncio

from ...domain.errors import RuntimeUnavailable
from ...domain.run import Intent
from ...domain.runtime import RuntimeKind
from ..provider import (
    RuntimeHealth,
)


class HealthMixin:
    """Runtime health and identity."""

    async def health(self) -> bool:
        """Public readiness probe (used by ``GET /health/ready``) — the same
        check ``execute`` makes before dispatching a turn, so readiness
        reflects what a real request would actually hit rather than merely
        naming a configured runtime kind."""
        try:
            health = await self._health()
        except RuntimeUnavailable:
            return False
        return health.healthy

    @property
    def missing_runtime_agents(self) -> tuple[str, ...]:
        """Named agents the last probe found absent from the runtime host.

        Read by the capability resolver, which is synchronous: the probe that
        fills this runs on readiness and before every dispatch, so the value is
        as fresh as the last time anything asked the runtime a question.
        """
        return self._missing_agents

    async def _health(self):
        try:
            health = await self._provider.health()
            self._missing_agents = tuple(getattr(health, "missing_agents", ()) or ())
            return health
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - adapter errors must stay typed
            raise RuntimeUnavailable(
                "the selected runtime could not be probed", reason=type(exc).__name__
            ) from exc

    async def _probe_health_with_retries(self) -> RuntimeHealth:
        """Pre-flight probe for ``execute`` — a fresh run and a recovery run
        both start here. A single point-in-time check races a runtime that
        was restarted moments earlier (progress log §3.8/§5.2): the process
        accepts TCP connections before it has finished loading its config, so
        the very next request after a restart can still see it as unhealthy.
        Retrying a few times, bounded by ``runtime_health_check_retries``,
        gives that window a chance to close without weakening the check —
        a runtime that is genuinely down still fails after the same attempts
        it always did.
        """
        attempts = max(1, self._settings.runtime_health_check_retries)
        delay = max(0.0, self._settings.runtime_health_check_retry_delay_s)
        health: RuntimeHealth | None = None
        for attempt in range(attempts):
            try:
                health = await self._health()
            except RuntimeUnavailable as exc:
                health = RuntimeHealth(healthy=False, detail=str(exc))
            if health.healthy or attempt == attempts - 1:
                return health
            await asyncio.sleep(delay)
        return health

    def _runtime_kind(self) -> RuntimeKind:
        try:
            kind = RuntimeKind(self._provider.kind)
        except ValueError as exc:
            raise RuntimeUnavailable(
                "the configured runtime adapter reports an unknown kind",
                kind=self._provider.kind,
            ) from exc
        if self._settings.kind != kind.value:
            raise RuntimeUnavailable(
                "the configured runtime kind does not match its injected adapter",
                configured=self._settings.kind,
                adapter=kind.value,
            )
        return kind

    def agent_for_capability(self, capability: str) -> str | None:
        return self._runtime_profiles.agent_for_capability(capability)

    def _runtime_version(self, kind: RuntimeKind) -> str:
        if kind is RuntimeKind.OPENCODE:
            return self._settings.opencode_version
        if kind is RuntimeKind.DSH:
            return self._settings.dsh_version
        return "in-process-scripted-v1"

    def _max_steps(self, intent: Intent) -> int:
        if intent is Intent.BUILD_REPORT:
            return self._settings.max_steps_report
        return (
            self._settings.max_steps_research
            if intent is Intent.DECISION_SUPPORT
            else self._settings.max_steps_qa
        )
