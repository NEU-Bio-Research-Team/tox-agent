"""Test helpers for exercising the MCP transport.

Two connection styles are used across the suite: an in-memory
``ClientSession`` for protocol-shape assertions that do not care about HTTP,
and a real Streamable HTTP round trip over ``httpx.ASGITransport`` for the
auth boundary — because the token is verified in the ASGI wrapper, not inside
the ``Server`` object itself.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.memory import create_connected_server_and_client_session


def connected_session(server):
    """In-memory: no HTTP, no auth boundary, just the MCP protocol."""
    return create_connected_server_and_client_session(server)


@asynccontextmanager
async def http_session(asgi_app, *, token: str | None):
    headers = {"authorization": f"Bearer {token}"} if token else None

    def factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=asgi_app), headers=headers, timeout=timeout
        )

    async with streamablehttp_client(
        "http://mcp.test/internal/mcp", headers=headers, httpx_client_factory=factory
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session
