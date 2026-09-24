"""A capability is reported because it was observed, not because HTTP was 200.

I14. The audit's own reproduction: a MockTransport returning nothing but
`data: [DONE]` made the old probe report `streaming=True, tool_calls=True,
structured_output=True`. No network request happened at all. An endpoint like
that was then advertised as READY for tool calling and JSON mode, and a run
depending on either failed mid-turn — after the user had been told it worked.

Each fixture below is one of the shapes that used to be certified.
"""
from __future__ import annotations

import json

import httpx
import pytest

from toxagent.connections.model import ModelConnection
from toxagent.connections.network import EgressPolicy
from toxagent.connections.probe import (
    OpenAICompatibleProbe,
    ProbeError,
    ProbeFailure,
    Support,
)
from toxagent.domain.runtime import AuthMode

pytestmark = pytest.mark.anyio

NOW = __import__("datetime").datetime(2026, 9, 9, tzinfo=__import__("datetime").timezone.utc)


def connection() -> ModelConnection:
    return ModelConnection.create(
        owner_id="owner-1", provider_id="openai_compatible", model_id="model-a",
        auth_mode=AuthMode.NONE, credential_ref=None,
        # A public host, so the destination policy is not what is under test
        # here; network policy has its own file.
        base_url="https://example.com/v1", now=NOW,
    )


def sse(*chunks: dict | str) -> bytes:
    lines = []
    for chunk in chunks:
        payload = chunk if isinstance(chunk, str) else json.dumps(chunk)
        lines.append(f"data: {payload}")
    return ("\n".join(lines) + "\n").encode()


def probe_with(body: bytes, status: int = 200) -> OpenAICompatibleProbe:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body, headers={"content-type": "text/event-stream"})

    return OpenAICompatibleProbe(
        egress=EgressPolicy.LOCAL, transport=httpx.MockTransport(handler)
    )


def delta(content: str | None = None, tool_calls: list | None = None) -> dict:
    piece: dict = {}
    if content is not None:
        piece["content"] = content
    if tool_calls is not None:
        piece["tool_calls"] = tool_calls
    return {"choices": [{"delta": piece}]}


# --- the exact fixture the audit reproduced ---------------------------------

async def test_a_done_only_stream_advertises_nothing():
    result = await probe_with(sse("[DONE]")).probe(connection(), None)
    assert result.streaming is not Support.SUPPORTED
    assert result.tool_calls is not Support.SUPPORTED
    assert result.structured_output is not Support.SUPPORTED
    # And it is unknown, not unsupported: nothing here proves absence either.
    assert result.tool_calls is Support.UNKNOWN


async def test_a_done_only_stream_stores_no_capability_as_true():
    result = await probe_with(sse("[DONE]")).probe(connection(), None)
    capabilities = result.to_capabilities()
    assert (capabilities.streaming, capabilities.tool_calls, capabilities.structured_output) == (
        False, False, False,
    )


# --- streaming ---------------------------------------------------------------

async def test_streaming_needs_a_real_content_delta():
    """A line beginning `data:` was the whole of the old check."""
    result = await probe_with(sse(delta("{"), delta('"ok": true}'), "[DONE]")).probe(
        connection(), None
    )
    assert result.streaming is Support.SUPPORTED


async def test_a_stream_of_empty_deltas_is_not_streaming():
    result = await probe_with(sse(delta(""), "[DONE]")).probe(connection(), None)
    assert result.streaming is Support.UNKNOWN


# --- structured output -------------------------------------------------------

async def test_structured_output_needs_content_that_parses_as_json():
    result = await probe_with(sse(delta('{"ok": true}'), "[DONE]")).probe(connection(), None)
    assert result.structured_output is Support.SUPPORTED


async def test_prose_is_not_structured_output():
    result = await probe_with(sse(delta("Sure, here you go!"), "[DONE]")).probe(
        connection(), None
    )
    assert result.structured_output is Support.UNKNOWN


async def test_json_in_a_fenced_block_still_counts():
    """Endpoints wrap JSON-mode output even when told not to; that is still
    structured output, and calling it unsupported would be its own wrong."""
    result = await probe_with(sse(delta('```json\n{"ok": true}\n```'), "[DONE]")).probe(
        connection(), None
    )
    assert result.structured_output is Support.SUPPORTED


# --- tool calls --------------------------------------------------------------

async def test_tool_calls_need_a_tool_call():
    call = [{"index": 0, "function": {"name": "probe_echo", "arguments": '{"word":"ok"}'}}]
    result = await probe_with(sse(delta(tool_calls=call), "[DONE]")).probe(connection(), None)
    assert result.tool_calls is Support.SUPPORTED


async def test_an_endpoint_that_ignored_the_tool_is_unknown_not_unsupported():
    """A model may simply decline to call one. Recording that as unsupported
    would be as wrong as the old unconditional True."""
    result = await probe_with(sse(delta('{"ok": true}'), "[DONE]")).probe(connection(), None)
    assert result.tool_calls is Support.UNKNOWN


# --- typed failures ----------------------------------------------------------

async def test_a_malformed_chunk_is_a_protocol_error_not_a_missing_capability():
    with pytest.raises(ProbeError) as excinfo:
        await probe_with(b"data: {not json\n").probe(connection(), None)
    assert excinfo.value.kind is ProbeFailure.PROTOCOL_ERROR


async def test_a_rejected_credential_is_reported_as_unauthorized():
    with pytest.raises(ProbeError) as excinfo:
        await probe_with(b"", status=401).probe(connection(), "wrong-key")
    assert excinfo.value.kind is ProbeFailure.UNAUTHORIZED


async def test_a_server_error_is_a_protocol_error():
    with pytest.raises(ProbeError) as excinfo:
        await probe_with(b"", status=500).probe(connection(), None)
    assert excinfo.value.kind is ProbeFailure.PROTOCOL_ERROR


async def test_an_empty_response_is_refused_rather_than_reported_as_capabilities():
    with pytest.raises(ProbeError) as excinfo:
        await probe_with(b"").probe(connection(), None)
    assert excinfo.value.kind is ProbeFailure.PROTOCOL_ERROR


async def test_a_redirect_is_refused_rather_than_followed():
    """Following one would step past the destination check (I15)."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    probe = OpenAICompatibleProbe(
        egress=EgressPolicy.LOCAL, transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProbeError) as excinfo:
        await probe.probe(connection(), None)
    assert excinfo.value.kind is ProbeFailure.PROTOCOL_ERROR


# --- no credential ever leaves in an error -----------------------------------

async def test_no_failure_message_contains_the_credential():
    secret = "sk-do-not-log-me"
    for body, status in ((b"", 401), (b"", 500), (b"data: {bad\n", 200), (b"", 200)):
        with pytest.raises(ProbeError) as excinfo:
            await probe_with(body, status=status).probe(connection(), secret)
        assert secret not in str(excinfo.value)
