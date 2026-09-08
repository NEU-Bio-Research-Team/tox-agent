"""The scripted runtime adapter (ADR 0004).

Not a model. It implements ``AgentRuntimeProvider`` in-process, with the
"model's" tool choices supplied by a plain Python function instead of a
provider call. It exists so the gateway, the binding lifecycle and the report
Q&A workflow can be exercised deterministically in CI without a pinned OpenCode
or DSH binary, a provider credential, or network egress — and so a frozen eval
fixture (plan section 16.3) is exactly one of these scripts.

A script receives a ``ScriptTurn`` and drives it by calling tools through the
*same* ``ToolRunner`` a real MCP connection would reach — visibility, budgets
and validation all apply exactly as they do to OpenCode or DSH. What is
skipped is only the transport: no HTTP, no MCP framing, because that path is
already covered on its own terms by ``tests/integration/test_mcp_server.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from ...domain.runtime import RuntimeCapabilities
from ...tools.registry import ToolContext, ToolRegistry
from ...tools.runner import ToolRunner
from ..provider import (
    RuntimeEvent,
    RuntimeEventType,
    RuntimeHealth,
    RuntimeReceipt,
    RuntimeSession,
    RuntimeSessionSpec,
    RuntimeTurn,
    CancelOutcome,
    CloseOutcome,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ScriptTurn:
    """What a script gets to work with. It may call tools and read the
    accumulated event log; it may not reach into the database or bypass the
    tool runner — anything it wants must go through a tool, same as a model."""

    user_message: str
    system_prompt: str
    tool_context: ToolContext
    _runner: ToolRunner
    events: list[RuntimeEvent] = field(default_factory=list)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.events.append(
            RuntimeEvent(
                RuntimeEventType.TOOL_REQUESTED, _now(), {"tool_name": name, "arguments": arguments}
            )
        )
        result = await self._runner.call(self.tool_context, name, arguments)
        self.events.append(
            RuntimeEvent(
                RuntimeEventType.TOOL_COMPLETED, _now(),
                {"tool_name": name, "status": result["status"]}, raw=result,
            )
        )
        return result

    def say(self, text: str) -> None:
        self.events.append(RuntimeEvent(RuntimeEventType.MESSAGE_DELTA, _now(), {"text": text}))

    def report_usage(
        self,
        tokens: dict[str, Any],
        *,
        cost: dict[str, Any] | None = None,
    ) -> None:
        """Emit a provider-shaped usage report for deterministic tests.

        This mirrors a runtime event rather than writing product storage
        directly, so W2-13's normalization path is exercised exactly as an
        OpenCode report is.
        """
        payload: dict[str, Any] = {"tokens": dict(tokens)}
        if cost is not None:
            payload["cost"] = dict(cost)
        self.events.append(RuntimeEvent(RuntimeEventType.USAGE_REPORTED, _now(), payload))


ScriptFn = Callable[[ScriptTurn], Awaitable[None]]


class ScriptedRuntimeProvider:
    """One instance per turn's worth of scripted behaviour. ``registry`` and
    ``runner`` are the same ones the MCP server would use; ``script`` stands in
    for a model's policy."""

    kind = "scripted"

    def __init__(self, registry: ToolRegistry, runner: ToolRunner, script: ScriptFn) -> None:
        self._registry = registry
        self._runner = runner
        self._script = script
        self._turns: dict[str, ScriptTurn] = {}
        self._session_contexts: dict[str, tuple[str, ToolContext]] = {}

    async def health(self) -> RuntimeHealth:
        return RuntimeHealth(healthy=True, detail="in-process, no external dependency")

    async def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            streaming=False, resume=False, cancel_turn=False, close_session=True,
            mcp_streamable_http=False, native_structured_output=True,
            usage=(), attachments=(),
        )

    async def create_session(self, spec: RuntimeSessionSpec) -> RuntimeSession:
        if spec.local_tool_context is None:
            raise ValueError(
                "the scripted adapter requires local_tool_context; it has no MCP transport "
                "of its own to authenticate a real one"
            )
        runtime_session_id = f"scripted-{spec.run_id}"
        self._session_contexts[runtime_session_id] = (spec.system_prompt, spec.local_tool_context)
        return RuntimeSession(
            runtime_session_id=runtime_session_id,
            provider_id=spec.provider_id, model_id=spec.model_id,
        )

    async def send(self, session: RuntimeSession, turn: RuntimeTurn) -> RuntimeReceipt:
        """Run the deterministic script through the ordinary provider contract.

        The local tool context was validated at ``create_session`` and is kept
        privately by this in-process adapter.  Real adapters use
        ``turn.capability_token`` to reach MCP; the scripted one intentionally
        does not parse or trust that token, because it never opens a transport.
        """
        try:
            system_prompt, tool_context = self._session_contexts[session.runtime_session_id]
        except KeyError as exc:
            raise ValueError("unknown scripted runtime session") from exc
        script_turn = ScriptTurn(turn.user_message, system_prompt, tool_context, self._runner)
        script_turn.events.append(
            RuntimeEvent(RuntimeEventType.TURN_STARTED, _now(), {"turn_id": turn.turn_id})
        )
        try:
            await self._script(script_turn)
            script_turn.events.append(
                RuntimeEvent(RuntimeEventType.TURN_IDLE, _now(), {"turn_id": turn.turn_id})
            )
        except Exception as exc:  # noqa: BLE001 — reported as a normalized failure, not raised
            script_turn.events.append(
                RuntimeEvent(
                    RuntimeEventType.TURN_FAILED, _now(),
                    {"turn_id": turn.turn_id, "reason": f"{type(exc).__name__}: {exc}"},
                )
            )
        self._turns[session.runtime_session_id] = script_turn
        return RuntimeReceipt(turn_id=turn.turn_id, accepted=True)

    async def events(self, session: RuntimeSession, after: str | None):
        """Replay this session's one completed scripted turn.

        Scripted sessions are deliberately single-turn.  This keeps frozen
        fixtures deterministic and prevents a test script from accidentally
        acquiring transcript state that a real provider did not expose.
        """
        turn = self._turns.get(session.runtime_session_id)
        if turn is None:
            return
        try:
            last_seen = int(after) if after is not None else -1
        except ValueError:
            last_seen = -1
        for index, event in enumerate(turn.events):
            if index <= last_seen:
                continue
            yield event

    async def cancel(self, session: RuntimeSession, receipt: RuntimeReceipt) -> CancelOutcome:
        # The script has always already finished by the time send() returns
        # (there is no concurrent worker to interrupt), so this is honest about
        # doing nothing rather than claiming a cancellation that cannot happen.
        return CancelOutcome(requested=True, runtime_cancel_supported=False, action="already_finished")

    async def close(self, session: RuntimeSession) -> CloseOutcome:
        self._session_contexts.pop(session.runtime_session_id, None)
        return CloseOutcome(closed=True)
