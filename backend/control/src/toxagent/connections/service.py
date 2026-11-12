"""Connection CRUD and explicit capability probing.

The probe itself lives in `probe.py`, the destination policy in `network.py`
and the list of providers with a working adapter in `providers.py`. Keeping
them apart is what makes each one testable: I14 was a probe that reported
capabilities it never checked, and it was written inline here where nothing
looked at it on its own.
"""
from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol

from ..domain.runtime import AuthMode
from . import providers
from .model import ConnectionStatus, ModelConnection
from .models import ModelCatalog, ModelOption
from .probe import ProbeError, ProbeFailure, ProtocolProbe, ProbeResult, Support
from .secrets import SecretStore, read_for_owner


class ConnectionNotFound(LookupError):
    pass


#: How long a model check is remembered. A working model costs a few tokens of
#: the user's own subscription to prove, so it is not proven on every click; a
#: refusal is remembered for less, because an account's plan can change.
#: ``busy`` / ``unavailable`` say nothing about the model and are never kept.
_CHECK_TTL_S = {"usable": 900.0, "unsupported": 300.0, "signin_rejected": 60.0}


class ModelProbe(Protocol):
    async def probe(self, connection: ModelConnection, credential: str | None) -> ProbeResult: ...


class ModelConnectionService:
    def __init__(
        self,
        database,
        secrets: SecretStore,
        probe: ModelProbe | None = None,
        *,
        catalog: ModelCatalog | None = None,
        hosts=None,
        agent_name: str = "toxagent",
    ) -> None:
        self._db, self._secrets = database, secrets
        self._probe = probe or ProtocolProbe()
        self._catalog = catalog or ModelCatalog()
        #: Private runtime hosts (harness/owner_hosts.py), when the
        #: deployment runs them. A ChatGPT sign-in is verified on one, and a
        #: deleted connection's host is stopped so its credential leaves disk.
        self._hosts = hosts
        self._agent_name = agent_name
        self._checks: dict[tuple[str, str], tuple[float, object]] = {}

    async def create(self, *, owner_id: str, provider_id: str, model_id: str,
                     auth_mode: AuthMode, base_url: str | None, credential: str | None,
                     display_name: str | None = None) -> ModelConnection:
        # Refuse a provider with no adapter here, with the reason, rather than
        # accepting the connection and failing every probe afterwards (I13).
        # `providers.get` raises UnsupportedProvider, which the API maps to a
        # 400 carrying the explanation.
        spec = providers.get(provider_id)
        # A provider with a known endpoint fills it in, so the field really is
        # optional for those and really is required for a self-hosted one.
        base_url = spec.resolve_base_url(base_url)
        if base_url is None:
            raise ValueError(
                f"{spec.display_name} has no default endpoint; supply a base URL"
            )
        if auth_mode not in spec.auth_modes:
            raise ValueError(
                f"{spec.display_name} does not support {auth_mode.value} authentication"
            )
        if auth_mode is AuthMode.CHATGPT_SUBSCRIPTION:
            # A sign-in is a token pair the server obtains, never a value a
            # browser pastes: it goes through the OAuth flow, which proves it.
            raise ValueError(
                "connect ChatGPT Plus/Pro with the sign-in flow, not by pasting a credential"
            )
        ref = None
        if credential:
            ref = self._secrets.put(owner_id, credential)
        try:
            connection = ModelConnection.create(
                owner_id=owner_id, provider_id=provider_id, model_id=model_id,
                auth_mode=auth_mode, credential_ref=ref, base_url=base_url,
                display_name=display_name,
                now=datetime.now(timezone.utc),
            )
            async with self._db.unit_of_work() as uow:
                await uow.model_connections.add(connection)
                await uow.commit()
            return connection
        except Exception:
            if ref:
                self._secrets.delete(ref)
            raise

    async def create_subscription(
        self, *, owner_id: str, model_id: str, credential: str, display_name: str | None = None
    ) -> ModelConnection:
        """Store a ChatGPT sign-in the OAuth flow just proved.

        Ready on creation: the device-code approval is the connection test,
        and a probe here would spend the user's quota to repeat it. What the
        model can do is not known until a run, so every capability is stored
        as not proven.
        """
        ref = self._secrets.put(owner_id, credential)
        try:
            now = datetime.now(timezone.utc)
            connection = replace(
                ModelConnection.create(
                    owner_id=owner_id, provider_id="openai", model_id=model_id,
                    auth_mode=AuthMode.CHATGPT_SUBSCRIPTION, credential_ref=ref,
                    base_url=None, display_name=display_name, now=now,
                ),
                status=ConnectionStatus.READY,
            )
            async with self._db.unit_of_work() as uow:
                await uow.model_connections.add(connection)
                await uow.commit()
            return connection
        except Exception:
            self._secrets.delete(ref)
            raise

    def _credential(self, item: ModelConnection) -> str | None:
        if not item.credential_ref:
            return None
        return read_for_owner(self._secrets, item.credential_ref, item.owner_id)

    async def list_models(
        self, *, provider_id: str, base_url: str | None, credential: str | None
    ) -> list[ModelOption]:
        """Models a not-yet-saved key can use, for the picker."""
        return await self._catalog.list(
            provider_id=provider_id, base_url=base_url, credential=credential
        )

    async def models_for(self, connection_id: str, *, owner_id: str) -> list[ModelOption]:
        """Models a saved connection's own credential can use."""
        item = await self.get(connection_id, owner_id=owner_id)
        credential = self._credential(item)
        if item.auth_mode is AuthMode.CHATGPT_SUBSCRIPTION:
            if self._hosts is None or credential is None:
                raise ProbeError(
                    ProbeFailure.PROTOCOL_ERROR,
                    "this deployment cannot reach the ChatGPT sign-in's runtime",
                )
            from .chatgpt import verify_subscription

            return await verify_subscription(self._hosts, item, credential)
        return await self._catalog.list(
            provider_id=item.provider_id, base_url=item.base_url, credential=credential
        )

    async def check_model(self, connection_id: str, *, owner_id: str, model_id: str):
        """Whether ``model_id`` really answers on this ChatGPT sign-in.

        The picker lists what the runtime and its catalogue name; this is the
        evidence that one of them works for this account, taken the way a run
        takes it, so a pick that stands is a pick that runs.
        """
        from .chatgpt import check_model

        item = await self.get(connection_id, owner_id=owner_id)
        if item.auth_mode is not AuthMode.CHATGPT_SUBSCRIPTION:
            raise ValueError(
                "only a ChatGPT sign-in is checked this way; a key's models are listed by "
                "its provider"
            )
        credential = self._credential(item)
        if self._hosts is None or credential is None:
            raise ProbeError(
                ProbeFailure.PROTOCOL_ERROR,
                "this deployment cannot reach the ChatGPT sign-in's runtime",
            )
        key = (connection_id, model_id.strip())
        hit = self._checks.get(key)
        if hit is not None and hit[0] > time.monotonic():
            return hit[1]
        result = await check_model(self._hosts, item, credential, model_id, agent=self._agent_name)
        for stale in [name for name, (expires, _) in self._checks.items() if expires <= time.monotonic()]:
            self._checks.pop(stale, None)
        ttl = _CHECK_TTL_S.get(result.status)
        if ttl is not None:
            self._checks[key] = (time.monotonic() + ttl, result)
        return result

    async def change_model(
        self, connection_id: str, *, owner_id: str, model_id: str,
        display_name: str | None = None,
    ) -> ModelConnection:
        """Point a connection at another model of the same account.

        The credential stays; the connection test does not carry over, since
        the new model may be one this account cannot use.
        """
        item = await self.get(connection_id, owner_id=owner_id)
        model_id = model_id.strip()
        if not model_id:
            raise ValueError("a model id is required")
        name = (display_name or "").strip()
        if not name:
            prefix = "ChatGPT" if item.auth_mode is AuthMode.CHATGPT_SUBSCRIPTION else item.provider_id
            name = f"{prefix} · {model_id}"
        updated = replace(
            item, model_id=model_id, display_name=name,
            status=ConnectionStatus.UNTESTED, updated_at=datetime.now(timezone.utc),
        )
        async with self._db.unit_of_work() as uow:
            await uow.model_connections.update_model(updated)
            await uow.commit()
        return updated

    async def list(self, *, owner_id: str):
        async with self._db.unit_of_work() as uow:
            return await uow.model_connections.list(owner_id=owner_id)

    async def get(self, connection_id: str, *, owner_id: str) -> ModelConnection:
        async with self._db.unit_of_work() as uow:
            item = await uow.model_connections.get(connection_id, owner_id=owner_id)
        if item is None:
            raise ConnectionNotFound(connection_id)
        return item

    async def test(self, connection_id: str, *, owner_id: str) -> ModelConnection:
        item = await self.get(connection_id, owner_id=owner_id)
        credential = self._credential(item)
        now = datetime.now(timezone.utc)
        try:
            if item.auth_mode is AuthMode.CHATGPT_SUBSCRIPTION:
                result = await self._test_subscription(item, credential)
            else:
                result = await self._probe.probe(item, credential)
            updated = replace(
                item, capabilities=result.to_capabilities(),
                status=ConnectionStatus.READY, updated_at=now,
            )
        except Exception:
            # The failed status is written whether the probe was refused, timed
            # out or answered wrongly; the caller re-raises to report which.
            updated = replace(item, status=ConnectionStatus.FAILED, updated_at=now)
            async with self._db.unit_of_work() as uow:
                await uow.model_connections.update_probe(updated)
                await uow.commit()
            raise
        async with self._db.unit_of_work() as uow:
            await uow.model_connections.update_probe(updated)
            await uow.commit()
        return updated

    async def _test_subscription(self, item: ModelConnection, credential: str | None) -> ProbeResult:
        if self._hosts is None or credential is None:
            raise ProbeError(
                ProbeFailure.PROTOCOL_ERROR,
                "this deployment cannot reach the ChatGPT sign-in's runtime",
            )
        from ..domain.errors import RuntimeUnavailable
        from .chatgpt import EnrollmentError, verify_subscription

        try:
            models = await verify_subscription(self._hosts, item, credential)
        except RuntimeUnavailable as exc:
            raise ProbeError(ProbeFailure.UNAUTHORIZED, str(exc)) from None
        except EnrollmentError as exc:
            raise ProbeError(ProbeFailure.UNREACHABLE, str(exc)) from None
        # A model id the runtime's catalog does not list may still be served
        # (the pinned binary predates newer models), so it is noted, not
        # refused: the sign-in is what this test can prove.
        listed = item.model_id in {model.id for model in models}
        return ProbeResult(
            Support.UNKNOWN, Support.UNKNOWN, Support.UNKNOWN,
            f"sign-in accepted by the runtime; {len(models)} model(s) available; "
            f"{item.model_id} {'listed' if listed else 'not in the runtime catalog'}",
        )

    async def delete(self, connection_id: str, *, owner_id: str) -> bool:
        item = await self.get(connection_id, owner_id=owner_id)
        async with self._db.unit_of_work() as uow:
            deleted = await uow.model_connections.delete(connection_id, owner_id=owner_id)
            await uow.commit()
        for name in [name for name in self._checks if name[0] == connection_id]:
            self._checks.pop(name, None)
        if deleted and self._hosts is not None:
            await self._hosts.forget_connection(connection_id)
        if deleted and item.credential_ref:
            self._secrets.delete(item.credential_ref)
        return deleted
