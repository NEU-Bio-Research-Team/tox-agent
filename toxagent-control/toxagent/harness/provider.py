"""The runtime provider contract (plan section 10.1).

Every adapter — the deterministic ``scripted`` one and the future OpenCode and
DSH ones — implements this and nothing else. The gateway (``harness.gateway``)
never reaches past it into an adapter's internals, which is what lets the
scripted adapter stand in for a real one in every test above this layer.

Capability is reported, never assumed from an adapter's name (plan section
10.1): ``capabilities()`` is what a caller trusts, and an adapter that cannot
actually cancel a turn says so rather than accepting a cancel it will ignore.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from ..domain.runtime import RuntimeCapabilities


class RuntimeEventType(str, Enum):
    """Plan section 10.2. An adapter maps whatever its own transport emits onto
    exactly these; anything it cannot map is logged and never presented as one
    of these, because guessing here is how a failed turn gets read as idle."""

    SESSION_CREATED = "runtime.session.created"
    TURN_ACCEPTED = "runtime.turn.accepted"
    TURN_STARTED = "runtime.turn.started"
    MESSAGE_DELTA = "runtime.message.delta"
    TOOL_REQUESTED = "runtime.tool.requested"
    TOOL_COMPLETED = "runtime.tool.completed"
    USAGE_REPORTED = "runtime.usage.reported"
    TURN_IDLE = "runtime.turn.idle"
    TURN_FAILED = "runtime.turn.failed"
    SESSION_LOST = "runtime.session.lost"


@dataclass(frozen=True)
class RuntimeHealth:
    healthy: bool
    detail: str = ""


@dataclass(frozen=True)
class RuntimeSessionSpec:
    """What a turn needs from the product side to start a runtime session.

    ``local_tool_context`` is read only by the in-process ``scripted`` adapter
    (harness/adapters/scripted.py).  It never crosses a process boundary.  An
    HTTP-based adapter instead receives its MCP capability when the first turn
    is sent, after the control plane has persisted the binding with the real
    runtime session id.

    This sequencing is deliberate.  A capability is scoped to a binding, but
    a binding cannot truthfully be stored until ``create_session`` returns the
    runtime's opaque id.  Passing a pre-binding token to ``create_session``
    would either need a fake session id in the audit row or leave the token
    unbound.  Neither is acceptable at the authorization boundary.
    """

    session_id: str
    run_id: str
    provider_id: str
    model_id: str
    profile: str
    system_prompt: str
    system_prompt_hash: str
    tool_schema: tuple[dict[str, Any], ...]
    tool_schema_hash: str
    mcp_url: str
    max_steps: int
    deadline_at: datetime
    local_tool_context: Any = None


@dataclass(frozen=True)
class RuntimeSession:
    runtime_session_id: str
    provider_id: str
    model_id: str


@dataclass(frozen=True)
class RuntimeTurn:
    turn_id: str
    user_message: str
    deadline_at: datetime
    #: Ephemeral, run-scoped MCP authority.  It is intentionally absent from
    #: RuntimeBinding and product events; only its JTI is persisted by the
    #: capability-token store.
    capability_token: str


@dataclass(frozen=True)
class RuntimeReceipt:
    """Plan section 12.2: a receipt confirms enqueue, not a final result. The
    gateway learns the outcome from the normalized event stream, and — for
    what actually matters — from the product's own database, never from the
    runtime's self-report alone."""

    turn_id: str
    accepted: bool
    detail: str = ""


@dataclass(frozen=True)
class RuntimeEvent:
    type: RuntimeEventType
    occurred_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass(frozen=True)
class CancelOutcome:
    requested: bool
    runtime_cancel_supported: bool
    action: str


@dataclass(frozen=True)
class CloseOutcome:
    closed: bool


@runtime_checkable
class AgentRuntimeProvider(Protocol):
    kind: str

    async def health(self) -> RuntimeHealth: ...
    async def capabilities(self) -> RuntimeCapabilities: ...
    async def create_session(self, spec: RuntimeSessionSpec) -> RuntimeSession: ...
    async def send(self, session: RuntimeSession, turn: RuntimeTurn) -> RuntimeReceipt: ...
    def events(
        self, session: RuntimeSession, after: str | None
    ) -> AsyncIterator[RuntimeEvent]: ...
    async def cancel(self, session: RuntimeSession, receipt: RuntimeReceipt) -> CancelOutcome: ...
    async def close(self, session: RuntimeSession) -> CloseOutcome: ...
