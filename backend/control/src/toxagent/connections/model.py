"""Provider/model connection is independent from the agent runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..domain.ids import CONNECTION, new_id, require_id
from ..domain.runtime import AuthMode


@dataclass(frozen=True, slots=True)
class ConnectionCapabilities:
    streaming: bool = False
    tool_calls: bool = False
    structured_output: bool = False
    context_size: int | None = None


class ConnectionStatus(str, Enum):
    UNTESTED = "untested"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ModelConnection:
    id: str
    owner_id: str
    provider_id: str
    model_id: str
    auth_mode: AuthMode
    credential_ref: str | None
    base_url: str | None
    capabilities: ConnectionCapabilities
    status: ConnectionStatus
    created_at: datetime
    updated_at: datetime
    display_name: str = ""

    @classmethod
    def create(cls, *, owner_id: str, provider_id: str, model_id: str,
               auth_mode: AuthMode, credential_ref: str | None, base_url: str | None,
               display_name: str | None = None,
               now: datetime) -> "ModelConnection":
        if auth_mode is AuthMode.API_KEY and not credential_ref:
            raise ValueError("api_key connections require a credential reference")
        if auth_mode in {AuthMode.LOCAL, AuthMode.NONE} and credential_ref:
            raise ValueError("local/none connections cannot carry credentials")
        return cls(new_id(CONNECTION), owner_id, provider_id, model_id, auth_mode,
                   credential_ref, base_url, ConnectionCapabilities(),
                   ConnectionStatus.UNTESTED, now, now,
                   (display_name or f"{provider_id} · {model_id}").strip())

    def __post_init__(self) -> None:
        require_id(self.id, CONNECTION, field="connection.id")
        if not self.owner_id or not self.provider_id or not self.model_id:
            raise ValueError("owner, provider and model are required")

    def public_dict(self) -> dict[str, object]:
        return {
            "connection_id": self.id, "provider_id": self.provider_id,
            "model_id": self.model_id, "display_name": self.display_name or f"{self.provider_id} · {self.model_id}", "auth_mode": self.auth_mode.value,
            "base_url": self.base_url, "has_credential": self.credential_ref is not None,
            "capabilities": {
                "streaming": self.capabilities.streaming,
                "tool_calls": self.capabilities.tool_calls,
                "structured_output": self.capabilities.structured_output,
                "context_size": self.capabilities.context_size,
            },
            "status": self.status.value,
        }
