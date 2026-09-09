"""Connection CRUD and explicit capability probing.

The probe itself lives in `probe.py`, the destination policy in `network.py`
and the list of providers with a working adapter in `providers.py`. Keeping
them apart is what makes each one testable: I14 was a probe that reported
capabilities it never checked, and it was written inline here where nothing
looked at it on its own.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol

from ..domain.runtime import AuthMode
from . import providers
from .model import ConnectionStatus, ModelConnection
from .probe import OpenAICompatibleProbe, ProbeError, ProbeResult
from .secrets import SecretStore


class ConnectionNotFound(LookupError):
    pass


class ModelProbe(Protocol):
    async def probe(self, connection: ModelConnection, credential: str | None) -> ProbeResult: ...


class ModelConnectionService:
    def __init__(self, database, secrets: SecretStore, probe: ModelProbe | None = None) -> None:
        self._db, self._secrets = database, secrets
        self._probe = probe or OpenAICompatibleProbe()

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
        credential = self._secrets.get(item.credential_ref) if item.credential_ref else None
        now = datetime.now(timezone.utc)
        try:
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

    async def delete(self, connection_id: str, *, owner_id: str) -> bool:
        item = await self.get(connection_id, owner_id=owner_id)
        async with self._db.unit_of_work() as uow:
            deleted = await uow.model_connections.delete(connection_id, owner_id=owner_id)
            await uow.commit()
        if deleted and item.credential_ref:
            self._secrets.delete(item.credential_ref)
        return deleted
