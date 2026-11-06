"""Which models a credential can use, so the user picks one instead of guessing.

Learned from G1's ``ai_models``: ask the provider's own model-listing endpoint
with the key the user is about to save (or has saved), filter to the models
that can hold a conversation, and cache the answer briefly under a digest of
(provider, endpoint, key) — so the cache never holds or reveals the key.

Listing is free on every provider here; no inference is spent to fill a
picker. A provider that refuses or fails is reported as a typed error for
that one list, never as a reason to fall back to some other account's key.

The destination policy is the probe's (I15): the URL is checked before the
request, redirects are not followed, and a hosted deployment cannot be made to
call a private address through a "base URL".
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from . import providers
from .network import BlockedDestination, EgressPolicy, check
from .probe import ANTHROPIC_VERSION

#: Same shape OpenCode accepts as a model id; anything else could never be
#: selected for a run.
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/@+-]*$")
MAX_MODELS = 500
CACHE_SECONDS = 600
LIST_TIMEOUT_S = 15.0

_OPENAI_CHAT = re.compile(r"^(gpt-|chatgpt-|o\d|codex)")
_OPENAI_EXCLUDED = (
    "audio",
    "dall-e",
    "embedding",
    "image",
    "instruct",
    "moderation",
    "realtime",
    "search",
    "transcribe",
    "tts",
    "whisper",
)


class ModelListError(RuntimeError):
    """Typed, redacted: never the key, never a resolved internal address."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class ModelOption:
    id: str
    name: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name}


def clean(items) -> list[ModelOption]:
    models: list[ModelOption] = []
    seen: set[str] = set()
    for model_id, name in items:
        if (
            isinstance(model_id, str)
            and len(model_id) <= 128
            and MODEL_ID.match(model_id)
            and model_id not in seen
        ):
            seen.add(model_id)
            models.append(
                ModelOption(model_id, name if isinstance(name, str) and name else model_id)
            )
    return models[:MAX_MODELS]


#: Listed by several providers next to their chat models, never usable for a run.
_NOT_CHAT = (
    "embedding", "embed", "whisper", "tts", "transcribe", "moderation", "guard",
    "imagen", "image", "veo", "aqa", "audio", "realtime",
)


def _chat_only(items):
    return [item for item in items if not any(w in str(item.get("id", "")).lower() for w in _NOT_CHAT)]


def _shape(provider_id: str, data: list[dict[str, Any]]) -> list[ModelOption]:
    if provider_id == "openai":
        chat = sorted(
            (
                item
                for item in data
                if _OPENAI_CHAT.match(str(item.get("id", "")))
                and not any(word in str(item.get("id")) for word in _OPENAI_EXCLUDED)
            ),
            key=lambda item: item.get("created") or 0,
            reverse=True,
        )
        return clean((item.get("id"), item.get("id")) for item in chat)
    if provider_id == "anthropic":
        return clean((item.get("id"), item.get("display_name")) for item in data)
    if provider_id == "openrouter":
        text = [
            item
            for item in data
            if "text" in ((item.get("architecture") or {}).get("output_modalities") or ["text"])
        ]
        return clean((item.get("id"), item.get("name")) for item in text)
    if provider_id == "google":
        # Gemini's OpenAI-compatible list names models "models/<id>"; OpenCode
        # and the native API want the bare id.
        return clean(
            (str(item.get("id", "")).removeprefix("models/"),
             item.get("display_name") or str(item.get("id", "")).removeprefix("models/"))
            for item in _chat_only(data)
        )
    if provider_id in {"groq", "mistral", "xai", "deepseek"}:
        return clean((item.get("id"), item.get("id")) for item in _chat_only(data))
    return clean((item.get("id"), item.get("id")) for item in data)


class ModelCatalog:
    def __init__(
        self,
        *,
        egress: EgressPolicy = EgressPolicy.LOCAL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._egress = egress
        self._transport = transport
        self._cache: dict[str, tuple[float, list[ModelOption]]] = {}

    async def list(
        self, *, provider_id: str, base_url: str | None, credential: str | None
    ) -> list[ModelOption]:
        spec = providers.get(provider_id)
        resolved = spec.resolve_base_url(base_url)
        if resolved is None:
            raise ModelListError("protocol_error", f"{spec.display_name} needs a base URL")
        if spec.models_path is None:
            raise ModelListError("unsupported", f"{spec.display_name} does not list models")
        key = hashlib.sha256(
            f"{provider_id}\0{resolved}\0{credential or ''}".encode("utf-8")
        ).hexdigest()
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
        models = await self._fetch(spec, resolved, credential)
        for stale in [item for item, value in self._cache.items() if value[0] <= now]:
            self._cache.pop(stale, None)
        self._cache[key] = (now + CACHE_SECONDS, models)
        return models

    async def _fetch(
        self, spec: providers.ProviderSpec, base_url: str, credential: str | None
    ) -> list[ModelOption]:
        url = base_url.rstrip("/") + (spec.models_path or "")
        if spec.id == "anthropic":
            url += "?limit=1000"
        try:
            check(url, self._egress)
        except BlockedDestination as exc:
            raise ModelListError("blocked", str(exc)) from None
        headers: dict[str, str] = {}
        if spec.protocol is providers.Protocol.ANTHROPIC_MESSAGES:
            headers["anthropic-version"] = ANTHROPIC_VERSION
            if credential:
                headers["x-api-key"] = credential
        elif credential:
            headers["Authorization"] = f"Bearer {credential}"
        try:
            async with httpx.AsyncClient(
                timeout=LIST_TIMEOUT_S,
                transport=self._transport,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = await client.get(url, headers=headers)
        except httpx.TimeoutException:
            raise ModelListError("timeout", f"no answer within {LIST_TIMEOUT_S:.0f}s") from None
        except httpx.HTTPError as exc:
            raise ModelListError(
                "unreachable", f"could not reach the endpoint: {type(exc).__name__}"
            ) from None
        if response.status_code in (401, 403):
            raise ModelListError("unauthorized", "the provider rejected this credential")
        if response.status_code == 429:
            raise ModelListError("rate_limited", "the provider's rate limit was reached")
        if response.status_code >= 300:
            raise ModelListError(
                "protocol_error", f"the provider answered HTTP {response.status_code}"
            )
        try:
            data = response.json().get("data") or []
        except (ValueError, AttributeError):
            raise ModelListError("protocol_error", "the provider returned no model list") from None
        if not isinstance(data, list):
            raise ModelListError("protocol_error", "the provider returned no model list")
        return _shape(spec.id, [item for item in data if isinstance(item, dict)])
