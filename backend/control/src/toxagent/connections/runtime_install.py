"""How one user AI connection is presented to an OpenCode runtime host.

Pure data and mapping, kept below the harness so the connection services (the
ChatGPT sign-in, the subscription test) and the runtime host pool share one
definition without the lower layer importing the upper one.

Verified live against the pinned OpenCode 1.17.11 before it was written:

- a provider in OpenCode's catalog takes its credential through
  ``PUT /auth/{provider}`` — ``{"type": "api", "key": ...}`` for a key, the
  OAuth token pair for a ChatGPT sign-in;
- a custom OpenAI-compatible endpoint does *not* read ``/auth``; its key must
  be ``options.apiKey``, given here as ``{env:TOXAGENT_PROVIDER_KEY}`` so it
  arrives through the host's environment rather than its config file;
- ``OPENCODE_CONFIG_CONTENT`` merges over ``OPENCODE_CONFIG`` (the agent
  profile), so a model the binary's catalog predates can be declared without
  losing the product agents.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from ..domain.errors import RuntimeProtocolError, RuntimeUnavailable
from . import providers

#: The env var a custom provider's key is read from inside its host.
PROVIDER_KEY_ENV = "TOXAGENT_PROVIDER_KEY"
OAUTH_FIELDS = {"type", "refresh", "access", "expires", "accountId", "enterpriseUrl"}


@dataclass(frozen=True)
class HostCredential:
    """Everything a host needs to act as one connection. Never logged."""

    connection_id: str
    owner_id: str
    provider_id: str
    model_id: str
    auth_mode: str
    base_url: str | None = None
    secret: str | None = field(default=None, repr=False)
    secret_ref: str | None = None

    def fingerprint(self) -> str:
        material = json.dumps(
            [self.provider_id, self.model_id, self.auth_mode, self.base_url, self.secret],
            separators=(",", ":"),
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RuntimeInstall:
    """How one connection is presented to OpenCode."""

    runtime_provider_id: str
    config: dict[str, Any]
    env: dict[str, str] = field(repr=False, default_factory=dict)
    auth: dict[str, Any] | None = field(repr=False, default=None)


def parse_oauth(secret: str | None) -> dict[str, Any]:
    """A ChatGPT subscription credential, validated to OpenCode's OAuth schema."""
    try:
        value = json.loads(secret or "")
    except ValueError as exc:
        raise RuntimeUnavailable(
            "the stored ChatGPT sign-in is not readable; reconnect it"
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("type") != "oauth"
        or not isinstance(value.get("refresh"), str)
        or not isinstance(value.get("access"), str)
        or not isinstance(value.get("expires"), int)
        or set(value) - OAUTH_FIELDS
    ):
        raise RuntimeUnavailable("the stored ChatGPT sign-in is not readable; reconnect it")
    return value


def install_for(credential: HostCredential) -> RuntimeInstall:
    """Map a product connection onto OpenCode's provider configuration."""
    spec = providers.get(credential.provider_id)
    model_entry = {credential.model_id: {"name": credential.model_id}}
    if credential.auth_mode == "chatgpt_subscription":
        if spec.runtime_provider_id != "openai":
            raise RuntimeProtocolError("a ChatGPT sign-in can only back an OpenAI connection")
        return RuntimeInstall(
            runtime_provider_id="openai",
            config={"provider": {"openai": {"models": model_entry}}},
            auth=parse_oauth(credential.secret),
        )
    if spec.runtime_provider_id is not None and not spec.uses_custom_endpoint(credential.base_url):
        provider_config: dict[str, Any] = {"models": model_entry}
        return RuntimeInstall(
            runtime_provider_id=spec.runtime_provider_id,
            config={"provider": {spec.runtime_provider_id: provider_config}},
            auth={"type": "api", "key": credential.secret} if credential.secret else None,
        )
    base_url = spec.resolve_base_url(credential.base_url)
    if not base_url:
        raise RuntimeProtocolError("this connection has no endpoint to install")
    options: dict[str, Any] = {"baseURL": base_url}
    env: dict[str, str] = {}
    if credential.secret:
        options["apiKey"] = "{env:" + PROVIDER_KEY_ENV + "}"
        env[PROVIDER_KEY_ENV] = credential.secret
    if spec.runtime_provider_id is not None:
        # A catalog provider pointed at a gateway: keep its own SDK, move it.
        return RuntimeInstall(
            runtime_provider_id=spec.runtime_provider_id,
            config={
                "provider": {spec.runtime_provider_id: {"options": options, "models": model_entry}}
            },
            env=env,
        )
    return RuntimeInstall(
        runtime_provider_id=providers.CUSTOM_RUNTIME_PROVIDER,
        config={
            "provider": {
                providers.CUSTOM_RUNTIME_PROVIDER: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": spec.display_name,
                    "options": options,
                    "models": model_entry,
                }
            }
        },
        env=env,
    )
