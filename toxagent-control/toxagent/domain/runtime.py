"""RuntimeBinding — which runtime answered, pinned hash by hash.

Plan section 5.8. The binding exists so that a run can be explained after the
fact: same question, same tools, different model — the audit says which. The
runtime's own session id appears here as a correlation key and nothing more;
authorisation comes from the capability token the control plane issues, never
from an identifier the runtime chose for itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any

from .ids import RUNTIME_BINDING, SESSION, new_id, require_id


class RuntimeKind(str, Enum):
    OPENCODE = "opencode"
    DSH = "dsh"
    SCRIPTED = "scripted"


class BindingStatus(str, Enum):
    ACTIVE = "active"
    LOST = "lost"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """Probed, never inferred from the runtime's name (plan section 10.1)."""

    streaming: bool = False
    resume: bool = False
    cancel_turn: bool = False
    close_session: bool = False
    mcp_streamable_http: bool = False
    native_structured_output: bool = False
    usage: tuple[str, ...] = ()
    attachments: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "streaming": self.streaming,
            "resume": self.resume,
            "cancel_turn": self.cancel_turn,
            "close_session": self.close_session,
            "mcp_streamable_http": self.mcp_streamable_http,
            "native_structured_output": self.native_structured_output,
            "usage": list(self.usage),
            "attachments": list(self.attachments),
        }


@dataclass(frozen=True, slots=True)
class RuntimeBinding:
    id: str
    session_id: str
    runtime_kind: RuntimeKind
    runtime_version: str
    runtime_session_id: str
    provider_id: str
    model_id: str
    profile_hash: str
    tool_schema_hash: str
    system_prompt_hash: str
    capabilities: RuntimeCapabilities
    status: BindingStatus
    created_at: datetime
    closed_at: datetime | None = None
    selection_reason: str = ""

    def __post_init__(self) -> None:
        require_id(self.id, RUNTIME_BINDING, field="binding.id")
        require_id(self.session_id, SESSION, field="binding.session_id")

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        runtime_kind: RuntimeKind,
        runtime_version: str,
        runtime_session_id: str,
        provider_id: str,
        model_id: str,
        profile_hash: str,
        tool_schema_hash: str,
        system_prompt_hash: str,
        capabilities: RuntimeCapabilities,
        now: datetime,
        selection_reason: str = "",
    ) -> "RuntimeBinding":
        return cls(
            id=new_id(RUNTIME_BINDING),
            session_id=session_id,
            runtime_kind=runtime_kind,
            runtime_version=runtime_version,
            runtime_session_id=runtime_session_id,
            provider_id=provider_id,
            model_id=model_id,
            profile_hash=profile_hash,
            tool_schema_hash=tool_schema_hash,
            system_prompt_hash=system_prompt_hash,
            capabilities=capabilities,
            status=BindingStatus.ACTIVE,
            created_at=now,
            selection_reason=selection_reason,
        )

    def lost(self, *, now: datetime) -> "RuntimeBinding":
        return replace(self, status=BindingStatus.LOST, closed_at=now)

    def closed(self, *, now: datetime) -> "RuntimeBinding":
        return replace(self, status=BindingStatus.CLOSED, closed_at=now)

    def manifest(self) -> dict[str, Any]:
        """The hashes that go into the run audit (plan section 15.1)."""
        return {
            "runtime_binding_id": self.id,
            "runtime_kind": self.runtime_kind.value,
            "runtime_version": self.runtime_version,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "profile_hash": self.profile_hash,
            "tool_schema_hash": self.tool_schema_hash,
            "system_prompt_hash": self.system_prompt_hash,
        }
