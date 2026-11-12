"""ChatGPT Plus/Pro sign-in, through OpenCode's own OAuth, for one owner at a time.

Learned from G1_Web_Design (``services/opencode_oauth.py``), adapted to how
this control plane runs OpenCode:

1. ``start`` launches a credential-less private host for the owner, asks it
   which OpenAI sign-in methods it offers (``GET /provider/auth``) and picks
   the *headless* (device-code) one by its label rather than a hard-coded
   index, then calls ``POST /provider/openai/oauth/authorize``. The user gets
   ``https://auth.openai.com/codex/device`` and a one-time code. The browser
   flow is not used: its redirect goes to a loopback port on the server, which
   a remote browser cannot reach.
2. A background task calls ``POST /provider/openai/oauth/callback``, which
   OpenCode holds open until the user approves the code (or it expires).
3. On success the token pair is read from that host's own ``auth.json``,
   validated against OpenCode's OAuth schema, and the models the sign-in may
   use are read from ``GET /config/providers``. The host is then stopped, so
   the pair exists only in memory until the user picks a model.
4. ``complete`` stores the pair encrypted, bound to the owner, as an
   ``openai`` connection with ``auth_mode=chatgpt_subscription``.

Availability is a deployment decision, and the default is conservative for
the same reason G1's is: OpenAI documents subscription sign-in for open-source
and local tools, and asks hosted, multi-user services to register first
(https://developers.openai.com/siwc/token-sharing-open-source). A local,
single-user deployment may offer it; a hosted one must opt in explicitly.
No token, code or URL is ever logged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from ..domain.errors import RuntimeUnavailable
from .models import MODEL_ID, ModelOption, clean, is_openai_chat_model
from .refusal import classify
from .runtime_install import HostCredential, parse_oauth

log = logging.getLogger("toxagent.connections.chatgpt")

DOCUMENTATION_URL = "https://developers.openai.com/siwc/token-sharing-open-source"
AUTH_HOST = "auth.openai.com"
#: How long a started sign-in waits for the user before it is abandoned.
ENROLLMENT_TIMEOUT_S = 900.0
#: How long a finished sign-in waits for the user to pick a model.
CONNECTED_TTL_S = 900.0


class ChatGPTUnavailable(RuntimeError):
    """This deployment does not offer ChatGPT sign-in, with the reason."""


class EnrollmentError(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "documentation_url": DOCUMENTATION_URL,
        }


def availability(*, mode: str, hosts_enabled: bool, egress_policy: str) -> Availability:
    """Whether this deployment may offer ChatGPT sign-in, and why (not)."""
    if mode == "off":
        return Availability(False, "ChatGPT sign-in is turned off on this deployment")
    if not hosts_enabled:
        return Availability(
            False,
            "this deployment runs no private AI runtime hosts, so a personal ChatGPT "
            "sign-in has nowhere to run (TOXAGENT_OWNER_RUNTIME_HOSTS)",
        )
    if mode == "hosted":
        return Availability(
            True, "ChatGPT Plus/Pro via OpenCode sign-in (hosted, operator opted in)"
        )
    if egress_policy != "local":
        return Availability(
            False,
            "a hosted deployment offers ChatGPT sign-in only when its operator opts in "
            "(TOXAGENT_CHATGPT_OAUTH=hosted) after registering with OpenAI",
        )
    return Availability(True, "ChatGPT Plus/Pro via OpenCode sign-in (local deployment)")


def validate_authorization(payload: Any) -> tuple[str, str]:
    """The sign-in URL and instructions OpenCode returned, or refuse them."""
    if not isinstance(payload, dict):
        raise EnrollmentError("protocol_error", "the runtime returned no sign-in link")
    url = payload.get("url")
    instructions = payload.get("instructions")
    parsed = urlsplit(url if isinstance(url, str) else "")
    if (
        parsed.scheme != "https"
        or parsed.hostname != AUTH_HOST
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise EnrollmentError("protocol_error", "the runtime returned an unexpected sign-in link")
    if not isinstance(instructions, str) or not instructions or len(instructions) > 2000:
        raise EnrollmentError("protocol_error", "the runtime returned unexpected instructions")
    if payload.get("method") != "auto":
        raise EnrollmentError(
            "protocol_error", "the runtime offered a sign-in this server cannot finish"
        )
    return url, instructions


def headless_method(methods: Any) -> int:
    """Index of OpenCode's device-code ChatGPT method, found by its label."""
    if not isinstance(methods, list):
        raise EnrollmentError("unsupported", "the runtime offers no OpenAI sign-in methods")
    for index, item in enumerate(methods):
        if (
            isinstance(item, dict)
            and item.get("type") == "oauth"
            and "headless" in str(item.get("label", "")).lower()
        ):
            return index
    raise EnrollmentError("unsupported", "the runtime offers no device-code ChatGPT sign-in")


def openai_models(payload: Any, *, availability: str | None = None) -> list[ModelOption]:
    providers = payload.get("providers") if isinstance(payload, dict) else None
    openai = next(
        (item for item in providers or [] if isinstance(item, dict) and item.get("id") == "openai"),
        {},
    )
    models = openai.get("models") if isinstance(openai, dict) else None
    if not isinstance(models, dict):
        return []
    return clean(
        ((key, value.get("name") if isinstance(value, dict) else key)
         for key, value in models.items()),
        availability=availability,
    )


def catalogue_models(payload: Any) -> list[ModelOption]:
    """The chat models the runtime's catalogue knows for OpenAI, newest first.

    ``/config/providers`` is the runtime's *own* view for this sign-in: the
    pinned binary filters it to the models it was built to allow, so a model
    released after it is absent however current its catalogue. ``/provider``
    is the catalogue itself, unfiltered. Both are claims about a model, not
    about the account, which is why a model is checked with a real turn
    (``check_model``) before the user keeps it.
    """
    entries = payload.get("all") if isinstance(payload, dict) else None
    openai = next(
        (item for item in entries or [] if isinstance(item, dict) and item.get("id") == "openai"),
        {},
    )
    models = openai.get("models") if isinstance(openai, dict) else None
    if not isinstance(models, dict):
        return []

    def released(item: tuple[str, Any]) -> str:
        value = item[1].get("release_date") if isinstance(item[1], dict) else None
        return value if isinstance(value, str) else ""

    ordered = sorted(models.items(), key=lambda item: (released(item), item[0]), reverse=True)
    return clean(
        ((key, value.get("name") if isinstance(value, dict) else key)
         for key, value in ordered if isinstance(key, str) and is_openai_chat_model(key)),
        availability="catalogue",
    )


def merge_models(offered: list[ModelOption], known: list[ModelOption]) -> list[ModelOption]:
    """What the runtime offers first, then what only the catalogue knows."""
    seen = {item.id for item in offered}
    return [*offered, *(item for item in known if item.id not in seen)]


async def host_models(host) -> list[ModelOption]:
    """Every model a private host can name for a ChatGPT sign-in."""
    params = {"directory": str(host.workspace)}
    response = await host.client.get("/config/providers", params=params, timeout=20)
    if response.status_code != 200:
        raise EnrollmentError("unavailable", "the runtime could not list the sign-in's models")
    offered = openai_models(response.json(), availability="runtime")
    try:
        known = await host.client.get("/provider", params=params, timeout=20)
        catalogue = catalogue_models(known.json()) if known.status_code == 200 else []
    except (httpx.HTTPError, ValueError):
        catalogue = []  # the runtime's own list is still a usable answer
    return merge_models(offered, catalogue)


@dataclass
class _Enrollment:
    owner_id: str
    host: Any
    method: int
    url: str
    instructions: str
    expires_at: float
    state: str = "pending"
    error: str | None = None
    credential: str | None = field(default=None, repr=False)
    models: list[ModelOption] = field(default_factory=list)
    task: asyncio.Task | None = None
    host_released: bool = False


class ChatGPTEnrollment:
    def __init__(
        self,
        pool,
        connections,
        availability: Availability,
        *,
        timeout_s: float = ENROLLMENT_TIMEOUT_S,
        clock=time.time,
    ) -> None:
        self._pool = pool
        self._connections = connections
        self.availability = availability
        self._timeout_s = timeout_s
        self._clock = clock
        self._enrollments: dict[str, _Enrollment] = {}
        self._lock = asyncio.Lock()

    def _require_available(self) -> None:
        if not self.availability.available or self._pool is None:
            raise ChatGPTUnavailable(self.availability.reason)

    async def start(self, owner_id: str) -> dict[str, Any]:
        self._require_available()
        async with self._lock:
            await self._drop(owner_id)
            host = await self._pool.enrollment_host(owner_id)
            try:
                methods = await host.client.get("/provider/auth")
                index = headless_method((methods.json() or {}).get("openai"))
                response = await host.client.post(
                    "/provider/openai/oauth/authorize", json={"method": index}, timeout=60
                )
                if response.status_code != 200:
                    raise EnrollmentError("unavailable", "the runtime could not start a sign-in")
                url, instructions = validate_authorization(response.json())
            except (httpx.HTTPError, ValueError) as exc:
                await self._pool.discard(host)
                raise EnrollmentError(
                    "unavailable", "the runtime could not start a sign-in"
                ) from exc
            except BaseException:
                await self._pool.discard(host)
                raise
            enrollment = _Enrollment(
                owner_id=owner_id,
                host=host,
                method=index,
                url=url,
                instructions=instructions,
                expires_at=self._clock() + self._timeout_s,
            )
            enrollment.task = asyncio.create_task(self._await_callback(enrollment))
            self._enrollments[owner_id] = enrollment
            return self._view(enrollment)

    async def _await_callback(self, enrollment: _Enrollment) -> None:
        host = enrollment.host
        try:
            response = await host.client.post(
                "/provider/openai/oauth/callback",
                json={"method": enrollment.method},
                timeout=httpx.Timeout(30.0, read=self._timeout_s),
            )
            if response.status_code != 200 or response.json() is not True:
                raise EnrollmentError("not_completed", "the ChatGPT sign-in was not completed")
            record = host.read_auth("openai")
            if not record or record.get("type") != "oauth":
                raise EnrollmentError("not_completed", "the ChatGPT sign-in returned no credential")
            credential = json.dumps(parse_oauth(json.dumps(record)), separators=(",", ":"))
            try:
                models = await host_models(host)
            except (httpx.HTTPError, ValueError, EnrollmentError):
                models = []
            enrollment.credential = credential
            enrollment.models = models
            enrollment.state = "connected"
            enrollment.expires_at = self._clock() + CONNECTED_TTL_S
        except asyncio.CancelledError:
            enrollment.state = "cancelled"
            raise
        except httpx.TimeoutException:
            enrollment.state, enrollment.error = (
                "expired",
                "the sign-in code expired before it was used",
            )
        except EnrollmentError as exc:
            enrollment.state, enrollment.error = "failed", str(exc)
        except Exception:  # noqa: BLE001 - reported as a typed failure, never with detail
            log.exception("ChatGPT sign-in failed")
            enrollment.state, enrollment.error = "failed", "the ChatGPT sign-in failed"
        finally:
            # The pair lives in memory only; its host and auth file go now.
            await self._release_host(enrollment)

    async def _release_host(self, enrollment: _Enrollment) -> None:
        if enrollment.host_released:
            return
        enrollment.host_released = True
        try:
            await self._pool.discard(enrollment.host)
        except Exception:  # noqa: BLE001
            log.exception("could not stop the sign-in host")

    def _view(self, enrollment: _Enrollment) -> dict[str, Any]:
        view: dict[str, Any] = {
            "state": enrollment.state,
            "expires_in_s": max(0, int(enrollment.expires_at - self._clock())),
        }
        if enrollment.state == "pending":
            view["verification_url"] = enrollment.url
            view["instructions"] = enrollment.instructions
        if enrollment.state == "connected":
            view["models"] = [model.to_dict() for model in enrollment.models]
        if enrollment.error:
            view["error"] = enrollment.error
        return view

    async def status(self, owner_id: str) -> dict[str, Any]:
        self._require_available()
        enrollment = self._enrollments.get(owner_id)
        if enrollment is None:
            return {"state": "none"}
        if self._clock() >= enrollment.expires_at and enrollment.state in {"pending", "connected"}:
            await self._drop(owner_id)
            return {"state": "expired", "error": "the sign-in expired; start again"}
        return self._view(enrollment)

    async def complete(self, owner_id: str, *, model_id: str, display_name: str | None = None):
        self._require_available()
        async with self._lock:
            enrollment = self._enrollments.get(owner_id)
            if enrollment is None or enrollment.state != "connected" or not enrollment.credential:
                raise EnrollmentError("not_connected", "finish the ChatGPT sign-in first")
            if self._clock() >= enrollment.expires_at:
                await self._drop(owner_id)
                raise EnrollmentError("expired", "the sign-in expired; start again")
            model_id = model_id.strip()
            if not MODEL_ID.match(model_id) or len(model_id) > 128:
                raise EnrollmentError("invalid_model", "that is not a valid model id")
            connection = await self._connections.create_subscription(
                owner_id=owner_id,
                model_id=model_id,
                credential=enrollment.credential,
                display_name=display_name or f"ChatGPT · {model_id}",
            )
            self._enrollments.pop(owner_id, None)
            return connection

    async def cancel(self, owner_id: str) -> None:
        async with self._lock:
            await self._drop(owner_id)

    async def _drop(self, owner_id: str) -> None:
        enrollment = self._enrollments.pop(owner_id, None)
        if enrollment is None:
            return
        enrollment.credential = None
        if enrollment.task is not None and not enrollment.task.done():
            enrollment.task.cancel()
            try:
                await enrollment.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        # A task cancelled before its first step never reaches its `finally`,
        # which left the sign-in host's OpenCode process running.
        await self._release_host(enrollment)

    async def aclose(self) -> None:
        for owner_id in list(self._enrollments):
            await self._drop(owner_id)


async def verify_subscription(pool, connection, credential: str) -> list[ModelOption]:
    """Prove a stored ChatGPT sign-in still works, on its own host.

    Starting the host installs the pair (a malformed or revoked one is refused
    there); the model list then shows what the sign-in may use. Releasing the
    host writes back a pair the runtime refreshed meanwhile.
    """
    host = await pool.acquire(
        HostCredential(
            connection_id=connection.id,
            owner_id=connection.owner_id,
            provider_id=connection.provider_id,
            model_id=connection.model_id,
            auth_mode=connection.auth_mode.value,
            base_url=None,
            secret=credential,
            secret_ref=connection.credential_ref,
        )
    )
    try:
        return await host_models(host)
    finally:
        await pool.release(host)


#: A turn this small answers in a few seconds and spends a handful of tokens.
CHECK_SYSTEM = "Reply with the single word OK."
CHECK_PROMPT = "ping"
CHECK_TIMEOUT_S = 75.0


@dataclass(frozen=True, slots=True)
class ModelCheck:
    """Whether one model answered a real turn on one ChatGPT sign-in.

    ``usable`` is the only status that lets a pick stand. ``unsupported`` is the
    provider saying this account cannot use the model; ``signin_rejected`` says
    the sign-in itself no longer works; ``busy`` and ``unavailable`` say nothing
    about the model (a run holds the connection, the runtime did not answer).
    """

    model_id: str
    status: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"model_id": self.model_id, "status": self.status, "message": self.message}


async def check_model(
    pool, connection, credential: str, model_id: str, *, agent: str = "toxagent"
) -> ModelCheck:
    """Ask ``model_id`` one tiny question on this sign-in's own host.

    The listing of models is a claim; this is the evidence. It takes the same
    path a run takes (the host, the sign-in, the Codex backend), so what it says
    is what a run will find. A refused model costs nothing and says why; a
    working one costs a few tokens of the user's own subscription, which is why
    the service remembers the answer for a while.
    """
    model_id = model_id.strip()
    if not MODEL_ID.match(model_id) or len(model_id) > 128:
        raise ValueError("that is not a valid model id")
    try:
        host = await pool.acquire(
            HostCredential(
                connection_id=connection.id,
                owner_id=connection.owner_id,
                provider_id=connection.provider_id,
                model_id=model_id,
                auth_mode=connection.auth_mode.value,
                base_url=None,
                secret=credential,
                secret_ref=connection.credential_ref,
            )
        )
    except RuntimeUnavailable as exc:
        return ModelCheck(model_id, "busy", str(exc))
    params = {"directory": str(host.workspace)}
    session_id: str | None = None
    try:
        created = await host.client.post("/session", params=params, json={"title": "model check"}, timeout=20)
        session_id = created.json().get("id") if created.status_code == 200 else None
        if not isinstance(session_id, str):
            return ModelCheck(model_id, "unavailable", "the runtime could not open a session to check the model")
        reply = await host.client.post(
            f"/session/{session_id}/message",
            params=params,
            json={
                "model": {"providerID": "openai", "modelID": model_id},
                "agent": agent,
                "system": CHECK_SYSTEM,
                "parts": [{"type": "text", "text": CHECK_PROMPT}],
            },
            timeout=CHECK_TIMEOUT_S,
        )
        if reply.status_code != 200:
            return ModelCheck(model_id, "unavailable", f"the runtime answered HTTP {reply.status_code}")
        response = reply.json()
        info = response.get("info") if isinstance(response, dict) else None
        if not isinstance(info, dict) or info.get("role") != "assistant":
            return ModelCheck(model_id, "unavailable", "the runtime returned an invalid answer")
        refusal = classify(info.get("error"))
        if refusal is None:
            parts = response.get("parts")
            if info.get("error") is not None or not isinstance(parts, list) or not any(
                isinstance(part, dict) and part.get("type") == "text"
                and isinstance(part.get("text"), str) and part["text"].strip()
                for part in parts
            ):
                return ModelCheck(model_id, "unavailable", "the runtime returned no answer")
            return ModelCheck(model_id, "usable", "the model answered")
        if refusal.kind == "model_not_supported":
            return ModelCheck(model_id, "unsupported", refusal.message)
        if refusal.kind == "signin_rejected":
            return ModelCheck(model_id, "signin_rejected", refusal.message)
        return ModelCheck(model_id, "unavailable", refusal.message)
    except httpx.TimeoutException:
        return ModelCheck(model_id, "unavailable", f"no answer within {CHECK_TIMEOUT_S:.0f}s")
    except (httpx.HTTPError, ValueError, AttributeError):
        return ModelCheck(model_id, "unavailable", "the runtime did not answer the check")
    finally:
        if session_id is not None:
            try:
                await host.client.delete(f"/session/{session_id}", params=params, timeout=10)
            except httpx.HTTPError:
                pass  # the host's directory goes with the host
        await pool.release(host)
