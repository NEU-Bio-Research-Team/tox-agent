"""The MCP transport (plan sections 8, 10.1, 11.3, 14.1).

MCP is an adapter over the typed registry, never a second source of truth: a
tool's schema and its visibility are read straight from ``ToolRegistry``, and
execution goes through the same ``ToolRunner`` the runner tests exercise
directly. Nothing here re-implements policy.

One MCP ``Server`` is built per HTTP connection, bound to the
``CapabilityClaims`` resolved from that connection's bearer token. That keeps
authorisation out of contextvars and out of the tool-call handler's control
flow entirely: ``list_tools`` can only ever enumerate what the token allows, and
``call_tool`` for anything else is refused before the registry is even
consulted — the two paths a model could use to discover a denied tool (listing
it, or calling it and reading the error) collapse to the same outcome.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from .. import __version__
from ..application.policy import Actor
from ..domain.errors import Unauthenticated
from . import envelope
from .capability import CapabilityClaims, CapabilityTokenService
from .registry import ToolContext, ToolRegistry
from .runner import ToolRunner

log = logging.getLogger("toxagent.mcp")

SERVER_NAME = "toxagent"


def build_server(registry: ToolRegistry, runner: ToolRunner, claims: CapabilityClaims) -> Server:
    """One MCP server instance, closed over one run's capability claims."""

    server: Server = Server(name=SERVER_NAME, version=__version__)
    allowed = frozenset(claims.allowed_tools) & frozenset(
        t.name for t in registry.visible_for(claims.profile)
    )

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=tool.name, title=tool.title, description=tool.description,
                inputSchema=tool.json_schema(),
            )
            for tool in registry.visible_for(claims.profile)
            if tool.name in allowed
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        if name not in allowed:
            # Same shape a denied-and-unknown tool gets from the runner itself
            # (PROD-06): a model probing for a distinguishing error learns
            # nothing either way.
            result = envelope.failed(
                call_id="", tool_name=name, code="tool_denied",
                message=f"{name} is not available to this run",
            )
        else:
            context = ToolContext(
                session_id=claims.session_id,
                run_id=claims.run_id,
                actor=Actor(subject_id=claims.subject_id, roles=claims.roles),
                profile=claims.profile,
                deadline_at=claims.expires_at,
                language=claims.language,
            )
            result = await runner.call(context, name, arguments)

        payload = envelope.model_payload(result)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
            structuredContent=payload,
            isError=envelope.is_error(result),
        )

    return server


def _bearer_token(scope: dict[str, Any]) -> str:
    headers = dict(scope.get("headers") or [])
    raw = headers.get(b"authorization", b"").decode("latin-1")
    scheme, _, token = raw.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise Unauthenticated("this MCP endpoint requires a run capability token")
    return token.strip()


def mcp_asgi_app(
    capability_tokens: CapabilityTokenService, registry: ToolRegistry, runner: ToolRunner
) -> Callable[[dict, Callable, Callable], Awaitable[None]]:
    """An ASGI callable, mountable under the control plane app.

    Authentication happens before anything MCP-shaped is touched: a request
    with no valid capability token gets a 401 and never reaches the tool
    surface, session manager included.
    """

    async def app(scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
            return
        if scope["type"] != "http":
            return

        try:
            token = _bearer_token(scope)
            claims = await capability_tokens.verify(token)
        except Unauthenticated as exc:
            await _reject(send, 401, "unauthenticated", str(exc))
            return

        if datetime.now(timezone.utc) >= claims.expires_at:
            await _reject(send, 401, "unauthenticated", "capability token has expired")
            return

        server = build_server(registry, runner, claims)
        # Stateless: one transport per HTTP request. A run's tool traffic is a
        # handful of short-lived calls, not a long-lived session worth pooling.
        manager = StreamableHTTPSessionManager(app=server, json_response=True, stateless=True)
        async with manager.run():
            await manager.handle_request(scope, receive, send)

    return app


async def _reject(send: Callable, status: int, code: str, message: str) -> None:
    body = json.dumps({"error": {"code": code, "message": message}}).encode("utf-8")
    await send(
        {
            "type": "http.response.start", "status": status,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})
