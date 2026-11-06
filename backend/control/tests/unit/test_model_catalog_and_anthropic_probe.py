"""Model listing for the picker, and the Anthropic Messages probe.

The listing is what lets a user pick a model their key can actually use
instead of typing an id (G1's single "Model AI" combobox). The probe is what
earns Anthropic its place in the provider list (I13's rule).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from toxagent.connections.models import ModelCatalog, ModelListError
from toxagent.connections.network import EgressPolicy
from toxagent.connections.probe import (
    AnthropicMessagesProbe,
    ProbeError,
    ProbeFailure,
    ProtocolProbe,
    Support,
)

pytestmark = pytest.mark.anyio


def _transport(handler, seen):
    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.MockTransport(wrapped)


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    """Every test host resolves to a public address; no real DNS."""
    import socket

    real = socket.getaddrinfo

    def resolve(host, port, *args, **kwargs):
        # Only the providers' names are faked; the test database (Postgres on
        # CI, reached as localhost) must still resolve for real.
        if host in {"localhost", "127.0.0.1", "::1"} or str(host).startswith("postgres"):
            return real(host, port, *args, **kwargs)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)


async def test_openai_listing_keeps_chat_models_newest_first_and_sends_the_key():
    seen: list[httpx.Request] = []
    data = {"data": [
        {"id": "gpt-old", "created": 1}, {"id": "gpt-new", "created": 9},
        {"id": "text-embedding-3-large", "created": 5}, {"id": "whisper-1", "created": 6},
        {"id": "gpt-4o-realtime-preview", "created": 7}, {"id": "o3", "created": 8},
    ]}
    catalog = ModelCatalog(transport=_transport(lambda r: httpx.Response(200, json=data), seen))
    models = await catalog.list(provider_id="openai", base_url=None, credential="sk-a")
    assert [m.id for m in models] == ["gpt-new", "o3", "gpt-old"]
    assert str(seen[0].url) == "https://api.openai.com/v1/models"
    assert seen[0].headers["authorization"] == "Bearer sk-a"


async def test_anthropic_listing_uses_its_own_headers_and_display_names():
    seen: list[httpx.Request] = []
    data = {"data": [{"id": "claude-x", "display_name": "Claude X"}]}
    catalog = ModelCatalog(transport=_transport(lambda r: httpx.Response(200, json=data), seen))
    models = await catalog.list(provider_id="anthropic", base_url=None, credential="sk-ant")
    assert [(m.id, m.name) for m in models] == [("claude-x", "Claude X")]
    assert seen[0].headers["x-api-key"] == "sk-ant"
    assert "authorization" not in seen[0].headers
    assert seen[0].url.params["limit"] == "1000"


async def test_a_listing_is_cached_without_holding_the_key():
    seen: list[httpx.Request] = []
    catalog = ModelCatalog(
        transport=_transport(lambda r: httpx.Response(200, json={"data": [{"id": "m"}]}), seen)
    )
    for _ in range(3):
        await catalog.list(provider_id="deepseek", base_url=None, credential="sk-secret")
    assert len(seen) == 1
    assert "sk-secret" not in json.dumps(list(catalog._cache))
    await catalog.list(provider_id="deepseek", base_url=None, credential="sk-other")
    assert len(seen) == 2


@pytest.mark.parametrize(
    ("status", "kind"), [(401, "unauthorized"), (403, "unauthorized"), (429, "rate_limited"),
                         (500, "protocol_error"), (302, "protocol_error")],
)
async def test_a_refusal_is_typed_and_never_echoes_the_key(status, kind):
    catalog = ModelCatalog(transport=_transport(lambda r: httpx.Response(status, json={}), []))
    with pytest.raises(ModelListError) as excinfo:
        await catalog.list(provider_id="openrouter", base_url=None, credential="sk-leak")
    assert excinfo.value.kind == kind
    assert "sk-leak" not in str(excinfo.value)


async def test_a_hosted_deployment_will_not_list_a_private_endpoint(monkeypatch):
    import socket

    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port))],
    )
    seen: list[httpx.Request] = []
    catalog = ModelCatalog(
        egress=EgressPolicy.HOSTED,
        transport=_transport(lambda r: httpx.Response(200, json={"data": []}), seen),
    )
    with pytest.raises(ModelListError) as excinfo:
        await catalog.list(provider_id="openai_compatible", base_url="http://internal/v1", credential=None)
    assert excinfo.value.kind == "blocked"
    assert seen == []


def _sse(*events) -> bytes:
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)


def _anthropic_connection():
    return SimpleNamespace(
        provider_id="anthropic", model_id="claude-x", base_url="https://api.anthropic.com/v1"
    )


async def test_the_anthropic_probe_needs_a_real_tool_use_block():
    seen: list[httpx.Request] = []
    body = _sse(
        {"type": "message_start"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "name": "probe_echo"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "message_stop"},
    )
    probe = AnthropicMessagesProbe(
        transport=_transport(lambda r: httpx.Response(200, content=body), seen)
    )
    result = await probe.probe(_anthropic_connection(), "sk-ant")
    assert result.tool_calls is Support.SUPPORTED
    assert seen[0].headers["x-api-key"] == "sk-ant"
    assert seen[0].headers["anthropic-version"] == "2023-06-01"
    assert json.loads(seen[0].content)["tools"][0]["input_schema"]["type"] == "object"


async def test_anthropic_prose_without_a_tool_call_is_unknown_not_supported():
    body = _sse(
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
    )
    probe = AnthropicMessagesProbe(transport=_transport(lambda r: httpx.Response(200, content=body), []))
    result = await probe.probe(_anthropic_connection(), "sk-ant")
    assert result.streaming is Support.SUPPORTED
    assert result.tool_calls is Support.UNKNOWN
    assert result.structured_output is Support.UNKNOWN


async def test_an_anthropic_rejection_is_unauthorized():
    probe = AnthropicMessagesProbe(transport=_transport(lambda r: httpx.Response(401), []))
    with pytest.raises(ProbeError) as excinfo:
        await probe.probe(_anthropic_connection(), "bad")
    assert excinfo.value.kind is ProbeFailure.UNAUTHORIZED


async def test_the_protocol_probe_routes_by_provider():
    seen: list[httpx.Request] = []
    body = _sse({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}})
    probe = ProtocolProbe(transport=_transport(lambda r: httpx.Response(200, content=body), seen))
    await probe.probe(_anthropic_connection(), "sk-ant")
    assert seen[0].url.path == "/v1/messages"
