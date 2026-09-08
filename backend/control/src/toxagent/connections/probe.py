"""Finding out what an endpoint can do, by making it do it.

I14: the previous probe returned `tool_calls=True, structured_output=True`
unconditionally after any 2xx, and set `streaming=True` for any line beginning
`data:`. An endpoint whose entire response was `data: [DONE]` was therefore
certified as supporting tool calling and JSON mode. A run that then depended
on either failed mid-turn, after the user had been told the connection was
ready — the worst place to discover it.

Three changes:

**Each capability is decided by its own evidence.** Streaming requires a real
content delta, not merely a line with the right prefix. Structured output
requires content that parses as JSON. Tool calling requires a tool call to
come back for a prompt that asks for one.

**Unknown is not unsupported.** An endpoint that returned no tool call for
this particular prompt may still support them; a model simply chose not to
call one. Recording that as `unsupported` would be as wrong as recording it as
supported, so it is `unknown` and the UI can say so.

**Failure is typed.** `unreachable`, `unauthorized`, `protocol_error` and
`blocked` are different problems with different fixes, and the message is
redacted — no credential, and no resolved internal address.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from .model import ConnectionCapabilities
from .network import BlockedDestination, EgressPolicy, check

#: How long a probe may take. A model endpoint that cannot answer a one-token
#: prompt in this long is not usable for a run either.
PROBE_TIMEOUT_S = 30.0

#: What a capability is known to be. The middle value is the one the previous
#: implementation had no way to express, so it guessed instead.
class Support(str, Enum):
    SUPPORTED = "supported"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class ProbeFailure(str, Enum):
    UNREACHABLE = "unreachable"
    UNAUTHORIZED = "unauthorized"
    PROTOCOL_ERROR = "protocol_error"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"


class ProbeError(RuntimeError):
    def __init__(self, kind: ProbeFailure, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """What was observed, with each capability's evidence separate."""

    streaming: Support
    structured_output: Support
    tool_calls: Support
    #: Free-text note for the operator: what the endpoint did, in one line.
    detail: str = ""

    def to_capabilities(self) -> ConnectionCapabilities:
        """The stored booleans.

        Only SUPPORTED becomes True. UNKNOWN stores False, because a stored
        `True` is what a run will rely on — the nuance lives in `detail` and
        in the typed result, not in a boolean that has to mean one thing.
        """
        return ConnectionCapabilities(
            streaming=self.streaming is Support.SUPPORTED,
            structured_output=self.structured_output is Support.SUPPORTED,
            tool_calls=self.tool_calls is Support.SUPPORTED,
            context_size=None,
        )


_PROMPT = (
    "Call the probe_echo tool with word set to ok. "
    'Then reply with exactly this JSON and nothing else: {"ok": true}'
)

_TOOL = {
    "type": "function",
    "function": {
        "name": "probe_echo",
        "description": "Protocol probe. Echoes one word back.",
        "parameters": {
            "type": "object",
            "properties": {"word": {"type": "string"}},
            "required": ["word"],
        },
    },
}


def _parse_stream(lines: list[str]) -> tuple[bool, str, bool]:
    """Return (saw a content delta, accumulated content, saw a tool call).

    Tolerant of the shapes real endpoints send: some omit `delta`, some send
    keep-alive comments, some end without `[DONE]`. What it will not do is
    treat the absence of evidence as evidence.
    """
    saw_delta = False
    saw_tool_call = False
    content: list[str] = []

    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            # A malformed chunk is a protocol problem, not a capability. Let
            # the caller see it as such rather than silently downgrading.
            raise ProbeError(
                ProbeFailure.PROTOCOL_ERROR,
                "the endpoint streamed a chunk that is not JSON",
            ) from None
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or choice.get("message") or {}
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                saw_delta = True
                content.append(piece)
            if delta.get("tool_calls"):
                saw_tool_call = True
    return saw_delta, "".join(content), saw_tool_call


def _looks_like_json(content: str) -> bool:
    text = content.strip()
    if not text:
        return False
    # Some endpoints wrap JSON mode output in a fenced block even when asked
    # not to; that is still structured output, so unwrap before judging.
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:].strip() if text.lower().startswith("json") else text.strip()
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


class OpenAICompatibleProbe:
    """One real streamed turn against `POST {base_url}/chat/completions`."""

    def __init__(
        self,
        *,
        egress: EgressPolicy = EgressPolicy.LOCAL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._egress = egress
        self._transport = transport

    async def probe(self, connection, credential: str | None) -> ProbeResult:
        if not connection.base_url:
            raise ProbeError(
                ProbeFailure.PROTOCOL_ERROR,
                "this connection has no base URL to probe; choose a provider with a "
                "default endpoint, or supply one",
            )
        url = connection.base_url.rstrip("/") + "/chat/completions"
        # Before the request, not after: a blocked destination must never be
        # contacted, so this cannot be a check on the response (I15).
        try:
            check(url, self._egress)
        except BlockedDestination as exc:
            raise ProbeError(ProbeFailure.BLOCKED, str(exc)) from None

        headers = {"Authorization": f"Bearer {credential}"} if credential else {}
        payload: dict[str, Any] = {
            "model": connection.model_id,
            "messages": [{"role": "user", "content": _PROMPT}],
            "stream": True,
            "response_format": {"type": "json_object"},
            "tools": [_TOOL],
        }

        lines: list[str] = []
        try:
            async with httpx.AsyncClient(
                timeout=PROBE_TIMEOUT_S,
                transport=self._transport,
                # Redirects are not followed: the destination check above
                # applies to one URL, and a 302 to a private address would
                # walk straight past it (I15).
                follow_redirects=False,
            ) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        raise ProbeError(
                            ProbeFailure.PROTOCOL_ERROR,
                            "the endpoint redirected; point the base URL at the final "
                            "endpoint instead",
                        )
                    if response.status_code in (401, 403):
                        raise ProbeError(
                            ProbeFailure.UNAUTHORIZED,
                            "the endpoint rejected this credential",
                        )
                    if response.status_code >= 400:
                        raise ProbeError(
                            ProbeFailure.PROTOCOL_ERROR,
                            f"the endpoint answered HTTP {response.status_code}",
                        )
                    async for line in response.aiter_lines():
                        lines.append(line)
        except httpx.TimeoutException:
            raise ProbeError(
                ProbeFailure.TIMEOUT, f"no answer within {PROBE_TIMEOUT_S:.0f}s"
            ) from None
        except httpx.HTTPError as exc:
            # `str(exc)` on httpx errors carries the URL but never the
            # Authorization header; the URL is the user's own input.
            raise ProbeError(
                ProbeFailure.UNREACHABLE, f"could not reach the endpoint: {type(exc).__name__}"
            ) from None

        saw_delta, content, saw_tool_call = _parse_stream(lines)
        if not lines:
            raise ProbeError(
                ProbeFailure.PROTOCOL_ERROR, "the endpoint returned an empty response"
            )

        streaming = Support.SUPPORTED if saw_delta else Support.UNKNOWN
        if saw_tool_call:
            tool_calls = Support.SUPPORTED
        elif content:
            # It answered, in words, a prompt that asked for a tool call.
            # Suggestive, not conclusive: models decline tool calls.
            tool_calls = Support.UNKNOWN
        else:
            tool_calls = Support.UNKNOWN
        structured = Support.SUPPORTED if _looks_like_json(content) else Support.UNKNOWN

        detail = (
            f"{len(lines)} stream line(s); "
            f"content {'seen' if saw_delta else 'absent'}; "
            f"tool call {'seen' if saw_tool_call else 'absent'}"
        )
        return ProbeResult(streaming, structured, tool_calls, detail)
