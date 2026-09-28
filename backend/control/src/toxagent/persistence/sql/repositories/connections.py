"""Model-provider connections."""
from __future__ import annotations

from typing import Sequence

from sqlalchemy import and_, delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from ....connections.model import (
    ConnectionCapabilities,
    ConnectionStatus,
    ModelConnection,
)
from ....domain.errors import Conflict
from ....domain.runtime import AuthMode
from ...schema import (
    model_connections,
)
from .. import mapping as m


class SqlModelConnectionStore:
    def __init__(self, conn: AsyncConnection) -> None:
        self._conn = conn

    async def add(self, connection: ModelConnection) -> None:
        await self._conn.execute(insert(model_connections).values(
            id=connection.id, owner_id=connection.owner_id, provider_id=connection.provider_id,
            model_id=connection.model_id, display_name=connection.display_name, base_url=connection.base_url,
            auth_mode=connection.auth_mode.value, credential_ref=connection.credential_ref,
            capabilities={
                "streaming": connection.capabilities.streaming,
                "tool_calls": connection.capabilities.tool_calls,
                "structured_output": connection.capabilities.structured_output,
                "context_size": connection.capabilities.context_size,
            }, status=connection.status.value, created_at=connection.created_at,
            updated_at=connection.updated_at,
        ))

    async def get(self, connection_id: str, *, owner_id: str) -> ModelConnection | None:
        row = (await self._conn.execute(select(model_connections).where(and_(
            model_connections.c.id == connection_id,
            model_connections.c.owner_id == owner_id,
        )))).mappings().first()
        if row is None:
            return None
        caps = row["capabilities"] or {}
        return ModelConnection(
            id=row["id"], owner_id=row["owner_id"], provider_id=row["provider_id"],
            model_id=row["model_id"], auth_mode=AuthMode(row["auth_mode"]),
            credential_ref=row["credential_ref"], base_url=row["base_url"],
            capabilities=ConnectionCapabilities(
                streaming=bool(caps.get("streaming")), tool_calls=bool(caps.get("tool_calls")),
                structured_output=bool(caps.get("structured_output")),
                context_size=caps.get("context_size"),
            ), status=ConnectionStatus(row["status"]), created_at=m.utc(row["created_at"]),
            updated_at=m.utc(row["updated_at"]), display_name=row.get("display_name") or "",
        )

    async def list(self, *, owner_id: str) -> Sequence[ModelConnection]:
        ids = (await self._conn.execute(select(model_connections.c.id).where(
            model_connections.c.owner_id == owner_id
        ).order_by(model_connections.c.created_at))).scalars().all()
        items = [await self.get(item, owner_id=owner_id) for item in ids]
        return [item for item in items if item is not None]

    async def update_probe(self, connection: ModelConnection) -> None:
        result = await self._conn.execute(update(model_connections).where(and_(
            model_connections.c.id == connection.id,
            model_connections.c.owner_id == connection.owner_id,
        )).values(
            capabilities={
                "streaming": connection.capabilities.streaming,
                "tool_calls": connection.capabilities.tool_calls,
                "structured_output": connection.capabilities.structured_output,
                "context_size": connection.capabilities.context_size,
            }, status=connection.status.value, updated_at=connection.updated_at,
        ))
        if result.rowcount == 0:
            raise Conflict("model connection does not exist", connection_id=connection.id)

    async def delete(self, connection_id: str, *, owner_id: str) -> bool:
        result = await self._conn.execute(delete(model_connections).where(and_(
            model_connections.c.id == connection_id,
            model_connections.c.owner_id == owner_id,
        )))
        return result.rowcount > 0
