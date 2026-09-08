"""Pinned OpenCode V1 HTTP/SSE adapter.

This is intentionally a thin transport adapter.  It neither implements an
agent loop nor trusts OpenCode text as product state: the model can only commit
through the normal MCP ``submit_grounded_answer`` tool.

The V1 wire contract used here is captured from the installed 1.17.11 SDK:

* ``POST /session`` creates the runtime-local session;
* ``POST /mcp`` and ``POST /mcp/{name}/connect`` install the run-scoped remote
  MCP authority *after* the product binding has been persisted;
* ``POST /session/{id}/prompt_async`` accepts a turn;
* ``GET /global/event`` is the global SSE feed; and
* ``POST /session/{id}/abort`` is the actual V1 abort operation.

OpenCode's MCP configuration is project-scoped rather than message-scoped.  A
deployment using this adapter therefore gives each ToxAgent worker an isolated
OpenCode project/data directory.  Reusing a shared OpenCode project would let a
short-lived capability header outlive one product run and is rejected by the
deployment profile/runbook rather than papered over in this adapter.
"""
from __future__ import annotations

import json
import logging
import shutil
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from ...config import RuntimeSettings
from ...domain.errors import RuntimeProtocolError, RuntimeUnavailable
from ...domain.runtime import RuntimeCapabilities
from ..provider import (
    CancelOutcome,
    CloseOutcome,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeHealth,
    RuntimeReceipt,
    RuntimeSession,
    RuntimeSessionSpec,
    RuntimeTurn,
)

log = logging.getLogger("toxagent.harness.opencode_v1")

OPENCODE_V1_PIN = "1.17.11"
MCP_NAME = "toxagent"
#: OpenCode has named a remote MCP tool both ``mcp_<server>_<tool>`` and, since
#: the naming that made the deny-all permission rule ``toxagent_*`` (see the
#: Phase 3 progress log §3.2), plain ``<server>_<tool>``. The event normalizer
#: strips whichever prefix the running binary emitted so a normalized
#: ``tool_name`` is always the bare tool ("get_analysis_slice").
MCP_TOOL_PREFIXES = (f"mcp_{MCP_NAME}_", f"{MCP_NAME}_")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class OpenCodeV1Provider:
    """Adapter for one private, pinned OpenCode V1 runtime host.

    ``client`` is injectable for the API-contract suite.  Production uses an
    ``httpx.AsyncClient`` owned by the adapter and the application lifespan
    closes it with :meth:`aclose`.
    """

    kind = "opencode"

    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if settings.opencode_version != OPENCODE_V1_PIN:
            raise ValueError(
                "OpenCodeV1Provider only supports the pinned "
                f"V1 contract {OPENCODE_V1_PIN}; got {settings.opencode_version!r}"
            )
        if not settings.opencode_directory:
            raise ValueError("TOXAGENT_OPENCODE_DIRECTORY is required for the V1 adapter")
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=settings.opencode_base_url,
            timeout=httpx.Timeout(settings.opencode_request_timeout_s),
        )
        self._owns_client = client is None
        self._specs: dict[str, RuntimeSessionSpec] = {}
        self._directories: dict[str, str] = {}
        self._locally_managed_directories: set[str] = set()

    def _query(self, directory: str | None = None) -> dict[str, str]:
        return {"directory": directory or self._settings.opencode_directory}

    def _directory_for_spec(self, spec: RuntimeSessionSpec) -> str:
        # ``run_id`` is server-generated and validated by the domain type, so
        # this cannot turn into a traversal path chosen by a user/model.
        return self._settings.opencode_directory.rstrip("/") + "/" + spec.run_id

    def _directory_for_session(self, runtime_session_id: str) -> str:
        directory = self._directories.get(runtime_session_id)
        if directory is None:
            raise RuntimeProtocolError("unknown OpenCode V1 runtime session")
        return directory

    def _create_local_directory(self, directory: str) -> None:
        """Create a run workspace only in the explicit single-host mode.

        The default remains supervisor-owned because a real runtime host can
        be remote from the control plane. Resolve and parent-check both paths
        before making anything so this convenience flag can never delete or
        create a directory selected by a model or browser.
        """
        if not self._settings.opencode_create_run_directories:
            return
        base = Path(self._settings.opencode_directory).resolve()
        target = Path(directory).resolve()
        if target.parent != base:
            raise RuntimeProtocolError("invalid local OpenCode run workspace")
        try:
            target.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise RuntimeUnavailable("could not create local OpenCode run workspace") from exc
        self._locally_managed_directories.add(str(target))

    def _remove_local_directory(self, directory: str) -> bool:
        if directory not in self._locally_managed_directories:
            return True
        self._locally_managed_directories.discard(directory)
        target = Path(directory).resolve()
        base = Path(self._settings.opencode_directory).resolve()
        if target.parent != base or target.is_symlink():
            log.error("refused to remove invalid local OpenCode run workspace")
            return False
        try:
            shutil.rmtree(target)
        except OSError:
            log.exception("could not remove local OpenCode run workspace")
            return False
        return True

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self) -> RuntimeHealth:
        """Probe the live server and verify the dedicated product agent exists."""
        try:
            response = await self._client.get("/agent", params=self._query())
        except httpx.HTTPError as exc:
            return RuntimeHealth(False, f"OpenCode V1 request failed: {type(exc).__name__}")
        if response.status_code != 200:
            return RuntimeHealth(False, f"OpenCode V1 /agent returned {response.status_code}")
        try:
            agents = response.json()
        except ValueError:
            return RuntimeHealth(False, "OpenCode V1 /agent returned non-JSON")
        if not isinstance(agents, list):
            return RuntimeHealth(False, "OpenCode V1 /agent returned an invalid shape")
        names = {
            item.get("name")
            for item in agents
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if self._settings.agent_name not in names:
            return RuntimeHealth(
                False,
                f"dedicated OpenCode agent {self._settings.agent_name!r} is unavailable",
            )
        return RuntimeHealth(True, f"OpenCode V1 {OPENCODE_V1_PIN} agent is available")

    async def capabilities(self) -> RuntimeCapabilities:
        # These are V1 API capabilities, verified by the contract suite.  V1
        # has no trustworthy event cursor/resume contract, so reconnects are
        # reconciled from product state rather than advertised as resume.
        return RuntimeCapabilities(
            streaming=True,
            resume=False,
            cancel_turn=True,
            close_session=True,
            mcp_streamable_http=True,
            native_structured_output=False,
            usage=("input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens"),
            attachments=("text",),
        )

    async def create_session(self, spec: RuntimeSessionSpec) -> RuntimeSession:
        directory = self._directory_for_spec(spec)
        self._create_local_directory(directory)
        try:
            response = await self._request(
                "POST",
                "/session",
                params=self._query(directory),
                json={"title": f"ToxAgent {spec.run_id}"},
                expected={200},
                operation="create runtime session",
            )
        except Exception:
            self._remove_local_directory(directory)
            raise
        payload = self._json_object(response, operation="create runtime session")
        session_id = payload.get("id")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeProtocolError("OpenCode V1 created a session without an id")
        self._specs[session_id] = spec
        self._directories[session_id] = directory
        return RuntimeSession(
            runtime_session_id=session_id,
            provider_id=spec.provider_id,
            model_id=spec.model_id,
        )

    async def send(self, session: RuntimeSession, turn: RuntimeTurn) -> RuntimeReceipt:
        spec = self._specs.get(session.runtime_session_id)
        if spec is None:
            raise RuntimeProtocolError("unknown OpenCode V1 runtime session")
        directory = self._directory_for_session(session.runtime_session_id)
        if not spec.mcp_url.startswith(("http://", "https://")):
            raise RuntimeProtocolError("OpenCode V1 requires an absolute private MCP URL")

        # The capability is deliberately absent from both the user prompt and
        # the model-visible schema.  The runtime management API installs it as
        # a private remote-MCP header only after the binding is durable.
        await self._install_run_mcp(spec, turn.capability_token, directory)
        # ``spec.max_steps`` (the gateway's per-intent step budget) is not sent
        # here: V1's prompt_async body has no step field at all (checked
        # against the pinned OpenAPI doc). The only enforced cap is the
        # checked-in agent profile's static ``maxSteps``, the same value for
        # every intent (agent_profiles/opencode/README.md, progress log §4.6).
        response = await self._request(
            "POST",
            f"/session/{session.runtime_session_id}/prompt_async",
            params=self._query(directory),
            json={
                "model": {"providerID": session.provider_id, "modelID": session.model_id},
                "agent": self._settings.agent_name,
                "system": spec.system_prompt,
                "parts": [{"type": "text", "text": turn.user_message}],
            },
            expected={204},
            operation="accept runtime turn",
        )
        del response  # Explicit: V1 returns 204; completion arrives only via SSE/state.
        return RuntimeReceipt(turn_id=turn.turn_id, accepted=True)

    async def _install_run_mcp(self, spec: RuntimeSessionSpec, token: str, directory: str) -> None:
        config = {
            "type": "remote",
            "url": spec.mcp_url,
            "enabled": True,
            "headers": {"Authorization": f"Bearer {token}"},
            "timeout": self._settings.opencode_mcp_timeout_ms,
        }
        await self._request(
            "POST",
            "/mcp",
            params=self._query(directory),
            json={"name": MCP_NAME, "config": config},
            expected={200},
            operation="configure run-scoped ToxAgent MCP",
        )
        connected = await self._request(
            "POST",
            f"/mcp/{MCP_NAME}/connect",
            params=self._query(directory),
            expected={200},
            operation="connect run-scoped ToxAgent MCP",
        )
        try:
            connected_value = connected.json()
        except ValueError as exc:
            raise RuntimeProtocolError("OpenCode V1 MCP connect returned non-JSON") from exc
        if connected_value is not True:
            raise RuntimeUnavailable("OpenCode V1 did not connect the ToxAgent MCP server")

    async def events(
        self, session: RuntimeSession, after: str | None
    ) -> AsyncIterator[RuntimeEvent]:
        """Normalize the V1 global SSE feed for one opaque runtime session.

        V1 has no event cursor.  ``after`` is intentionally ignored and the
        terminal reconciliation below prevents a clean SSE close from being
        mistaken for success while a session is still busy.
        """
        del after
        saw_terminal = False
        turn_started = False
        try:
            # Do not preflight ``/session/status`` here.  ``prompt_async``
            # returns once the turn is queued, while V1 can still report the
            # previous idle state for a short scheduling window.  Treating
            # that stale idle result as terminal loses the entire turn before
            # the global event stream has a chance to report ``busy``.  This
            # was observed live with Luna: only the echoed user part and two
            # zero-token events arrived before the product incorrectly closed
            # the runtime binding.
            #
            # The post-stream reconciliation below remains necessary for a
            # cleanly closed SSE connection; unlike this preflight it runs
            # only after the stream itself has ended.
            # The management-API request timeout must not apply to the event
            # stream: a model turn is routinely silent for longer than one
            # request budget, and a read timeout there would be reported as a
            # lost session rather than as the ordinary wait it is. The turn is
            # already bounded by the run deadline in the gateway.
            stream_timeout = httpx.Timeout(
                self._settings.opencode_request_timeout_s, read=None
            )
            async with self._client.stream(
                "GET", "/global/event", timeout=stream_timeout
            ) as response:
                if response.status_code != 200:
                    yield RuntimeEvent(
                        RuntimeEventType.SESSION_LOST,
                        _now(),
                        {"reason": f"OpenCode V1 event endpoint returned {response.status_code}"},
                    )
                    return
                async for raw_payload in self._sse_payloads(response):
                    event = self._normalize_event(raw_payload, session.runtime_session_id)
                    if event is None:
                        continue
                    if event.type is RuntimeEventType.TURN_STARTED:
                        turn_started = True
                    # A global V1 feed can replay the user's just-created
                    # text part while the queued turn is still idle.  It is
                    # not assistant output and must not be presented as a
                    # diagnostic "model reply".  After the matching busy
                    # transition, text deltas belong to this turn normally.
                    if event.type is RuntimeEventType.MESSAGE_DELTA and not turn_started:
                        continue
                    yield event
                    if event.type in {
                        RuntimeEventType.TURN_IDLE,
                        RuntimeEventType.TURN_FAILED,
                        RuntimeEventType.SESSION_LOST,
                    }:
                        saw_terminal = True
                        return
        except httpx.HTTPError as exc:
            yield RuntimeEvent(
                RuntimeEventType.SESSION_LOST,
                _now(),
                {"reason": f"OpenCode V1 event transport failed: {type(exc).__name__}"},
            )
            return

        if saw_terminal:
            return
        if await self._is_idle(session.runtime_session_id):
            yield RuntimeEvent(RuntimeEventType.TURN_IDLE, _now(), {"reconciled": True})
        else:
            yield RuntimeEvent(
                RuntimeEventType.SESSION_LOST,
                _now(),
                {"reason": "OpenCode V1 event stream ended while the session was not idle"},
            )

    async def _sse_payloads(self, response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        data_lines: list[str] = []
        async for line in response.aiter_lines():
            if not line:
                if data_lines:
                    raw = "\n".join(data_lines)
                    data_lines.clear()
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        log.warning("ignored malformed OpenCode V1 SSE payload")
                        continue
                    if isinstance(parsed, dict):
                        yield parsed
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            try:
                parsed = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                return
            if isinstance(parsed, dict):
                yield parsed

    def _normalize_event(
        self, envelope: dict[str, Any], runtime_session_id: str
    ) -> RuntimeEvent | None:
        raw = envelope.get("payload")
        if not isinstance(raw, dict):
            log.debug("ignored unknown OpenCode V1 event envelope", extra={"event": envelope})
            return None
        event_type = raw.get("type")
        properties = raw.get("properties")
        if not isinstance(event_type, str) or not isinstance(properties, dict):
            return None

        event_session_id = self._event_session_id(properties)
        if event_session_id != runtime_session_id:
            return None

        if event_type == "session.status":
            status = properties.get("status")
            if isinstance(status, dict) and status.get("type") == "busy":
                return RuntimeEvent(RuntimeEventType.TURN_STARTED, _now(), {}, raw=envelope)
            if isinstance(status, dict) and status.get("type") == "idle":
                return RuntimeEvent(RuntimeEventType.TURN_IDLE, _now(), {}, raw=envelope)
        if event_type == "session.idle":
            return RuntimeEvent(RuntimeEventType.TURN_IDLE, _now(), {}, raw=envelope)
        if event_type == "session.error":
            return RuntimeEvent(RuntimeEventType.TURN_FAILED, _now(), dict(properties), raw=envelope)
        if event_type == "message.part.updated":
            part = properties.get("part")
            if not isinstance(part, dict):
                return None
            if part.get("type") == "text":
                text = properties.get("delta") or part.get("text")
                if isinstance(text, str) and text:
                    return RuntimeEvent(
                        RuntimeEventType.MESSAGE_DELTA, _now(), {"text": text}, raw=envelope
                    )
            if part.get("type") == "tool":
                return self._tool_event(part, envelope)
            if part.get("type") == "step-finish":
                tokens = part.get("tokens")
                if isinstance(tokens, dict):
                    return RuntimeEvent(
                        RuntimeEventType.USAGE_REPORTED, _now(), {"tokens": tokens}, raw=envelope
                    )
        if event_type == "message.updated":
            info = properties.get("info")
            if isinstance(info, dict) and info.get("role") == "assistant":
                tokens = info.get("tokens")
                if isinstance(tokens, dict):
                    return RuntimeEvent(
                        RuntimeEventType.USAGE_REPORTED, _now(), {"tokens": tokens}, raw=envelope
                    )
        log.debug("ignored unmapped OpenCode V1 event", extra={"event_type": event_type})
        return None

    @staticmethod
    def _event_session_id(properties: dict[str, Any]) -> str | None:
        direct = properties.get("sessionID")
        if isinstance(direct, str):
            return direct
        for key in ("part", "info"):
            nested = properties.get(key)
            if isinstance(nested, dict) and isinstance(nested.get("sessionID"), str):
                return nested["sessionID"]
        return None

    @staticmethod
    def _tool_event(part: dict[str, Any], raw: dict[str, Any]) -> RuntimeEvent | None:
        state = part.get("state")
        tool = part.get("tool")
        if not isinstance(state, dict) or not isinstance(tool, str):
            return None
        tool_name = tool
        for prefix in MCP_TOOL_PREFIXES:
            if tool_name.startswith(prefix):
                tool_name = tool_name[len(prefix):]
                break
        payload = {"tool_name": tool_name, "call_id": part.get("callID", "")}
        if state.get("status") in {"pending", "running"}:
            return RuntimeEvent(RuntimeEventType.TOOL_REQUESTED, _now(), payload, raw=raw)
        if state.get("status") in {"completed", "error"}:
            payload["status"] = state["status"]
            return RuntimeEvent(RuntimeEventType.TOOL_COMPLETED, _now(), payload, raw=raw)
        return None

    async def _is_idle(self, runtime_session_id: str) -> bool:
        try:
            response = await self._client.get(
                "/session/status", params=self._query(self._directory_for_session(runtime_session_id))
            )
            if response.status_code != 200:
                return False
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return False
        status = payload.get(runtime_session_id) if isinstance(payload, dict) else None
        return isinstance(status, dict) and status.get("type") == "idle"

    async def cancel(self, session: RuntimeSession, receipt: RuntimeReceipt) -> CancelOutcome:
        del receipt
        try:
            response = await self._request(
                "POST",
                f"/session/{session.runtime_session_id}/abort",
                params=self._query(self._directory_for_session(session.runtime_session_id)),
                expected={200},
                operation="abort runtime turn",
            )
            aborted = response.json()
        except (RuntimeProtocolError, RuntimeUnavailable, ValueError):
            return CancelOutcome(True, True, "abort_requested_outcome_unknown")
        return CancelOutcome(
            requested=True,
            runtime_cancel_supported=True,
            action="runtime_turn_aborted" if aborted is True else "abort_requested_outcome_unknown",
        )

    async def close(self, session: RuntimeSession) -> CloseOutcome:
        # Disconnect first so a private run capability is no longer usable by
        # this runtime.  Disconnect failure is captured by the false outcome;
        # the product binding will be marked lost rather than reusable.
        disconnected = False
        deleted_ok = False
        directory: str | None = None
        try:
            directory = self._directory_for_session(session.runtime_session_id)
            response = await self._request(
                "POST",
                f"/mcp/{MCP_NAME}/disconnect",
                params=self._query(directory),
                expected={200},
                operation="disconnect run-scoped ToxAgent MCP",
            )
            disconnected = response.json() is True
            deleted = await self._request(
                "DELETE",
                f"/session/{session.runtime_session_id}",
                params=self._query(directory),
                expected={200},
                operation="close runtime session",
            )
            deleted_ok = deleted.json() is True
        except (RuntimeProtocolError, RuntimeUnavailable, ValueError):
            pass
        finally:
            self._specs.pop(session.runtime_session_id, None)
            self._directories.pop(session.runtime_session_id, None)
        local_workspace_removed = (
            self._remove_local_directory(directory) if directory is not None else True
        )
        return CloseOutcome(closed=disconnected and deleted_ok and local_workspace_removed)

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, str],
        expected: set[int],
        operation: str,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        try:
            response = await self._client.request(method, url, params=params, json=json)
        except httpx.HTTPError as exc:
            raise RuntimeUnavailable(f"OpenCode V1 could not {operation}") from exc
        if response.status_code in expected:
            return response
        details = {"http_status": response.status_code}
        if response.status_code >= 500:
            raise RuntimeUnavailable(f"OpenCode V1 could not {operation}", **details)
        raise RuntimeProtocolError(f"OpenCode V1 rejected {operation}", **details)

    @staticmethod
    def _json_object(response: httpx.Response, *, operation: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeProtocolError(f"OpenCode V1 {operation} returned non-JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeProtocolError(f"OpenCode V1 {operation} returned an invalid JSON shape")
        return payload
