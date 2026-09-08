"""Which providers this control plane can actually talk to.

I13: the Settings page offered OpenAI, Anthropic and Google Gemini with an
empty default base URL, described the field as optional, and then failed every
capability probe with `capability probing requires an explicit base_url`. A
user could complete the form correctly and never get a working connection.

Two separate things were conflated. *Which providers exist* is a product
question; *which protocols this code speaks* is an engineering one, and only
the second may decide what the UI lists. Everything here speaks the
OpenAI-compatible chat/completions protocol, because that is the only adapter
that exists. Anthropic's Messages API and Gemini's generateContent are
different wire formats: listing them without an adapter is what produced a
form that could not succeed.

Adding a provider means adding an entry here *and* a probe that has been run
against it. A `default_base_url` of None means the deployment must supply one —
a self-hosted endpoint has no default anyone could guess.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..domain.runtime import AuthMode


class Protocol(str, Enum):
    """The wire format an adapter speaks."""

    #: `POST {base_url}/chat/completions`, SSE streaming, `tools` and
    #: `response_format` as OpenAI defines them.
    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    id: str
    display_name: str
    protocol: Protocol
    #: None when the deployment must supply one (a local or self-hosted
    #: endpoint). A provider with a fixed public endpoint names it, so the UI
    #: can prefill and the user need not know it.
    default_base_url: str | None
    auth_modes: tuple[AuthMode, ...]
    #: Shown next to the provider in the UI. Written for someone deciding
    #: whether to pick it, not as an apology.
    note: str = ""

    def resolve_base_url(self, supplied: str | None) -> str | None:
        """What this connection will actually call."""
        return (supplied or "").strip() or self.default_base_url


SUPPORTED: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="openai",
        display_name="OpenAI",
        protocol=Protocol.OPENAI_CHAT_COMPLETIONS,
        default_base_url="https://api.openai.com/v1",
        auth_modes=(AuthMode.API_KEY,),
    ),
    ProviderSpec(
        id="openrouter",
        display_name="OpenRouter",
        protocol=Protocol.OPENAI_CHAT_COMPLETIONS,
        default_base_url="https://openrouter.ai/api/v1",
        auth_modes=(AuthMode.API_KEY,),
    ),
    ProviderSpec(
        id="openai_compatible",
        display_name="OpenAI-compatible / self-hosted",
        protocol=Protocol.OPENAI_CHAT_COMPLETIONS,
        default_base_url=None,
        auth_modes=(AuthMode.API_KEY, AuthMode.NONE, AuthMode.LOCAL),
        note="Any server speaking OpenAI's chat/completions protocol. Base URL required.",
    ),
)

#: Named but not offered, with the reason. Kept here rather than deleted so
#: the answer to "why can't I pick Anthropic?" is in the code that decides it,
#: and so re-adding one is a visible change.
UNSUPPORTED: dict[str, str] = {
    "anthropic": (
        "Anthropic's Messages API is a different wire format from "
        "chat/completions and this control plane has no adapter for it yet. "
        "An Anthropic model reachable through an OpenAI-compatible gateway "
        "can be added as OpenAI-compatible."
    ),
    "gemini": (
        "Gemini's generateContent API is a different wire format from "
        "chat/completions and this control plane has no adapter for it yet. "
        "Gemini through an OpenAI-compatible gateway can be added as "
        "OpenAI-compatible."
    ),
}

_BY_ID = {spec.id: spec for spec in SUPPORTED}


class UnsupportedProvider(ValueError):
    """Named provider has no adapter. Carries the reason for the user."""

    def __init__(self, provider_id: str, reason: str) -> None:
        super().__init__(reason)
        self.provider_id = provider_id
        self.reason = reason


def get(provider_id: str) -> ProviderSpec:
    spec = _BY_ID.get(provider_id)
    if spec is not None:
        return spec
    reason = UNSUPPORTED.get(
        provider_id,
        f"unknown provider {provider_id!r}; supported: {sorted(_BY_ID)}",
    )
    raise UnsupportedProvider(provider_id, reason)


def catalogue() -> list[dict[str, object]]:
    """What the Settings page lists. Supported providers only."""
    return [
        {
            "provider_id": spec.id,
            "display_name": spec.display_name,
            "protocol": spec.protocol.value,
            "default_base_url": spec.default_base_url,
            "base_url_required": spec.default_base_url is None,
            "auth_modes": [mode.value for mode in spec.auth_modes],
            "note": spec.note,
        }
        for spec in SUPPORTED
    ]
