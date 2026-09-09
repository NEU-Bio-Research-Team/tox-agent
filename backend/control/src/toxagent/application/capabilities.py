"""One answer to "can this deployment actually do X".

Three places used to answer that question separately and disagree (I01, I02,
I05): the readiness endpoint named the *configured* runtime kind, admission
gated `evidence_research` on a research provider existing, and the scheduler
was the only one that knew whether a handler had ever been registered. A
request could pass admission, create a message and a run, and then fail on
"no handler is registered" — an orphaned run for work the deployment was
never able to do.

Two distinctions this module keeps, because collapsing either is what caused
those bugs:

**Configured is not available.** Compose declared `TOXAGENT_RUNTIME_KIND=scripted`
while the app only ever constructs a provider for `opencode`, so a stack
advertised a runtime it would never start. `configured` records intent;
`available` records what a request would actually meet.

**Mode is not failure.** A predictor-only deployment has no agent runtime *on
purpose*. It reports the conversational capabilities as unavailable with a
reason and stays `ready`; it must not report itself broken for a feature that
was deliberately not installed.

Availability comes in two costs. `intent(...)` and `snapshot()` are cheap and
synchronous — registry lookups only — and are what admission uses on every
request. `probe()` additionally makes live calls to the predictor, the
database, OCR and the runtime, and is what readiness uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

from ..domain.run import Intent


class DeploymentMode(str, Enum):
    """What this deployment was assembled to do.

    Derived from what is actually wired up, never read from an environment
    variable — the variable is what lied.
    """

    #: Prediction, batch prediction and (when OCR is configured) structure
    #: recognition. No agent runtime, so no report Q&A, attribution or
    #: literature research. A complete, supportable product; not a degraded
    #: agent deployment.
    PREDICTOR_ONLY = "predictor_only"
    #: The above plus an agent runtime bound and its conversational intents
    #: registered.
    AGENT_ENABLED = "agent_enabled"


#: Intents a browser can ask for. `CLARIFICATION_REQUIRED` and `OUT_OF_SCOPE`
#: are router outcomes, not capabilities, and are deliberately absent.
REQUESTABLE: tuple[Intent, ...] = (
    Intent.ANALYSIS,
    Intent.ANALYSIS_BATCH,
    Intent.STRUCTURE_RECOGNITION,
    Intent.REPORT_QA,
    Intent.ATTRIBUTION,
    Intent.EVIDENCE_RESEARCH,
)

#: The intents that need an agent runtime turn.
CONVERSATIONAL: frozenset[Intent] = frozenset(
    {Intent.REPORT_QA, Intent.ATTRIBUTION, Intent.EVIDENCE_RESEARCH}
)


@dataclass(frozen=True)
class Capability:
    """Whether one capability can be used, and why not when it cannot.

    `reason` is written for the person who has to fix it, and is safe to show
    a user: it names the missing dependency, never a stack trace or a secret.
    """

    name: str
    configured: bool
    available: bool
    checked_at: datetime
    reason: str | None = None
    #: Extra facts for operators (the probed dependency's own report). Never
    #: consulted for a decision — `available` is the decision.
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "configured": self.configured,
            "available": self.available,
            "checked_at": self.checked_at.isoformat(),
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.detail:
            payload["detail"] = self.detail
        return payload


class _Scheduler(Protocol):
    def handles(self, intent: Intent) -> bool: ...


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CapabilityResolver:
    """The single source both admission and readiness consult.

    Holds no state of its own beyond references: the scheduler gains handlers
    during startup *after* the services that ask about them are constructed,
    so every answer is computed on demand rather than captured once.
    """

    def __init__(
        self,
        *,
        scheduler: _Scheduler,
        runtime_kind: str,
        research_provider: object | None = None,
        ocr_client: object | None = None,
        runtime_gateway_getter: Any = None,
    ) -> None:
        self._scheduler = scheduler
        self._runtime_kind = runtime_kind
        self._research_provider = research_provider
        self._ocr = ocr_client
        # A callable rather than the gateway, because api/app.py binds the
        # gateway after this resolver exists.
        self._gateway_getter = runtime_gateway_getter or (lambda: None)

    # -- cheap, synchronous: what admission uses ---------------------------

    @property
    def mode(self) -> DeploymentMode:
        gateway_bound = self._gateway_getter() is not None
        registered = any(self._scheduler.handles(i) for i in CONVERSATIONAL)
        if gateway_bound and registered:
            return DeploymentMode.AGENT_ENABLED
        return DeploymentMode.PREDICTOR_ONLY

    def intent(self, intent: Intent) -> Capability:
        """Can a request for this intent reach a handler that can fulfil it?

        This is the check that was missing. A registered handler is necessary
        and, for the conversational intents, so is a bound runtime gateway:
        registering a handler that immediately fails is not a capability.
        """
        # `configured` is what this deployment's wiring declared: a handler
        # was registered for the intent. It is deliberately not read from
        # TOXAGENT_RUNTIME_KIND — that variable is what lied (I01).
        registered = self._scheduler.handles(intent)
        configured = registered
        reason: str | None = None
        available = registered

        if not registered:
            reason = _UNREGISTERED_REASON.get(intent, "no handler is registered for this intent")

        if intent in CONVERSATIONAL and registered and self._gateway_getter() is None:
            # Declared but unservable. Not a mode — incoherent wiring, and
            # readiness treats it as such.
            available = False
            reason = (
                "a handler is registered for this intent but no agent runtime "
                "is bound behind it"
            )

        if intent is Intent.EVIDENCE_RESEARCH and available and self._research_provider is None:
            available = False
            reason = "no literature provider is configured for this deployment"

        if intent is Intent.STRUCTURE_RECOGNITION and available and self._ocr is None:
            available = False
            reason = "no structure-recognition service is configured for this deployment"

        return Capability(
            name=intent.value,
            configured=configured,
            available=available,
            checked_at=_now(),
            reason=reason,
        )

    def available(self, intent: Intent) -> bool:
        return self.intent(intent).available

    def snapshot(self) -> dict[str, Capability]:
        return {i.value: self.intent(i) for i in REQUESTABLE}

    def misconfigured(self) -> list[Capability]:
        """Capabilities this deployment declared and cannot serve.

        Distinct from a mode: a predictor-only stack declares no
        conversational handler and is healthy. A stack that registered one
        with no runtime behind it is broken, and readiness says so.
        """
        return [
            capability
            for intent in CONVERSATIONAL
            if (capability := self.intent(intent)).configured
            and not capability.available
        ]


#: Why an intent has no handler, phrased as the deployment fact it is rather
#: than as an internal registry miss.
_UNREGISTERED_REASON: dict[Intent, str] = {
    Intent.STRUCTURE_RECOGNITION: (
        "no structure-recognition service is configured for this deployment"
    ),
    Intent.REPORT_QA: (
        "this deployment has no agent runtime bound, so questions about an "
        "analysis cannot be answered here"
    ),
    Intent.ATTRIBUTION: (
        "this deployment has no agent runtime bound, so attribution cannot run here"
    ),
    Intent.EVIDENCE_RESEARCH: (
        "this deployment has no agent runtime bound, so literature research "
        "cannot run here"
    ),
}
