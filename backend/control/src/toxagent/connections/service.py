"""Connection CRUD and explicit capability probing."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol

import httpx

from ..domain.runtime import AuthMode
from .model import ConnectionCapabilities, ConnectionStatus, ModelConnection
from .secrets import SecretStore


class ConnectionNotFound(LookupError):
    pass


class ModelProbe(Protocol):
    async def probe(self, connection: ModelConnection, credential: str | None) -> ConnectionCapabilities: ...


class OpenAICompatibleProbe:
    """Probe declared protocol features with a real, minimal streamed turn."""

    async def probe(self, connection: ModelConnection, credential: str | None) -> ConnectionCapabilities:
        if not connection.base_url:
            raise ValueError("capability probing requires an explicit base_url")
        headers = {"Authorization": f"Bearer {credential}"} if credential else {}
        payload = {
            "model": connection.model_id,
            "messages": [{"role": "user", "content": "Return JSON: {\"ok\":true}"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "response_format": {"type": "json_object"},
            "tools": [{
                "type": "function",
                "function": {"name": "probe_noop", "description": "Protocol probe only",
                             "parameters": {"type": "object", "properties": {}}},
            }],
        }
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream(
                "POST", connection.base_url.rstrip("/") + "/chat/completions",
                headers=headers, json=payload,
            ) as response:
                response.raise_for_status()
                saw_chunk = False
                async for line in response.aiter_lines():
                    saw_chunk = saw_chunk or line.startswith("data:")
        # These booleans record acceptance by the live endpoint, not a guess
        # based on provider_id. Context size remains unknown unless an adapter
        # can measure or retrieve it explicitly.
        return ConnectionCapabilities(
            streaming=saw_chunk, tool_calls=True, structured_output=True, context_size=None,
        )


class ModelConnectionService:
    def __init__(self, database, secrets: SecretStore, probe: ModelProbe | None = None) -> None:
        self._db, self._secrets = database, secrets
        self._probe = probe or OpenAICompatibleProbe()

    async def create(self, *, owner_id: str, provider_id: str, model_id: str,
                     auth_mode: AuthMode, base_url: str | None, credential: str | None,
                     display_name: str | None = None) -> ModelConnection:
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
            capabilities = await self._probe.probe(item, credential)
            updated = replace(item, capabilities=capabilities, status=ConnectionStatus.READY, updated_at=now)
        except Exception:
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
