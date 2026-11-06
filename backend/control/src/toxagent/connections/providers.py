"""Which providers this control plane can actually talk to.

I13: the Settings page offered OpenAI, Anthropic and Google Gemini with an
empty default base URL, described the field as optional, and then failed every
capability probe with `capability probing requires an explicit base_url`. A
user could complete the form correctly and never get a working connection.

Two separate things were conflated. *Which providers exist* is a product
question; *which protocols this code speaks* is an engineering one, and only
the second may decide what the UI lists. A provider is listed when two things
exist for it: a probe that speaks its wire format (``probe.py``), and a way to
hand its credential to the agent runtime (``harness/owner_hosts.py``
maps ``runtime_provider_id`` onto OpenCode's own provider of that name).

Anthropic was refused here until a Messages-API probe existed; it now does.
Gemini still has none, and stays refused with the reason.

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
    #: `POST {base_url}/messages`, `x-api-key` + `anthropic-version`, SSE with
    #: typed `content_block_*` events.
    ANTHROPIC_MESSAGES = "anthropic_messages"


#: The custom OpenCode provider an OpenAI-compatible connection is installed
#: as on its own runtime host. Not a catalog name, so it can never collide
#: with a provider OpenCode already knows.
CUSTOM_RUNTIME_PROVIDER = "toxagent-byo"


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
    #: OpenCode's provider id for this one. None means the runtime installs it
    #: as the custom OpenAI-compatible provider pointed at the base URL.
    runtime_provider_id: str | None = None
    #: Path under the base URL that lists models, or None when the provider
    #: has no such endpoint worth calling.
    models_path: str | None = "/models"

    def resolve_base_url(self, supplied: str | None) -> str | None:
        """What this connection will actually call."""
        return (supplied or "").strip() or self.default_base_url

    def uses_custom_endpoint(self, base_url: str | None) -> bool:
        """True when the runtime must be pointed somewhere other than its default."""
        if self.runtime_provider_id is None:
            return True
        resolved = self.resolve_base_url(base_url)
        return bool(resolved) and resolved.rstrip("/") != (self.default_base_url or "").rstrip("/")


SUPPORTED: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="openai",
        display_name="OpenAI",
        protocol=Protocol.OPENAI_CHAT_COMPLETIONS,
        default_base_url="https://api.openai.com/v1",
        auth_modes=(AuthMode.API_KEY, AuthMode.CHATGPT_SUBSCRIPTION),
        note=(
            "API key (billed to the key's organisation) or a ChatGPT Plus/Pro sign-in. "
            "A custom base URL must serve OpenAI's Responses API; for a chat/completions-only "
            "gateway choose OpenAI-compatible."
        ),
        runtime_provider_id="openai",
    ),
    ProviderSpec(
        id="anthropic",
        display_name="Claude (Anthropic)",
        protocol=Protocol.ANTHROPIC_MESSAGES,
        default_base_url="https://api.anthropic.com/v1",
        auth_modes=(AuthMode.API_KEY,),
        note="Anthropic API key. A Claude Pro/Max sign-in cannot be used by other apps.",
        runtime_provider_id="anthropic",
    ),
    ProviderSpec(
        id="openrouter",
        display_name="OpenRouter",
        protocol=Protocol.OPENAI_CHAT_COMPLETIONS,
        default_base_url="https://openrouter.ai/api/v1",
        auth_modes=(AuthMode.API_KEY,),
        runtime_provider_id="openrouter",
    ),
    ProviderSpec(
        id="deepseek",
        display_name="DeepSeek",
        protocol=Protocol.OPENAI_CHAT_COMPLETIONS,
        default_base_url="https://api.deepseek.com/v1",
        auth_modes=(AuthMode.API_KEY,),
        runtime_provider_id="deepseek",
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
#: the answer to "why can't I pick Gemini?" is in the code that decides it,
#: and so re-adding one is a visible change.
UNSUPPORTED: dict[str, str] = {
    "gemini": (
        "Gemini's generateContent API is a different wire format from "
        "chat/completions and this control plane has no adapter for it yet. "
        "Gemini through an OpenAI-compatible gateway can be added as "
        "OpenAI-compatible."
    ),
    "claude_pro": (
        "A Claude.ai Pro/Max sign-in is not offered: Anthropic does not allow "
        "third-party apps to use a consumer subscription for inference. Use an "
        "Anthropic API key instead."
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


def catalogue(*, chatgpt: dict[str, object] | None = None) -> list[dict[str, object]]:
    """What the Settings page lists. Supported providers only.

    ``chatgpt`` is this deployment's ChatGPT sign-in availability
    (``{"available": bool, "reason": str}``). A provider's
    ``chatgpt_subscription`` mode is listed only while it is available, so the
    form never offers a sign-in the server would refuse.
    """
    entries: list[dict[str, object]] = []
    for spec in SUPPORTED:
        modes = [mode for mode in spec.auth_modes]
        oauth = None
        if AuthMode.CHATGPT_SUBSCRIPTION in modes:
            oauth = dict(chatgpt or {"available": False, "reason": "not configured"})
            if not oauth.get("available"):
                modes = [mode for mode in modes if mode is not AuthMode.CHATGPT_SUBSCRIPTION]
        entries.append(
            {
                "provider_id": spec.id,
                "display_name": spec.display_name,
                "protocol": spec.protocol.value,
                "default_base_url": spec.default_base_url,
                "base_url_required": spec.default_base_url is None,
                "auth_modes": [mode.value for mode in modes],
                "note": spec.note,
                "lists_models": spec.models_path is not None,
                "chatgpt_oauth": oauth,
            }
        )
    return entries
