"""Compatibility export of the stable AgentRuntime contract."""
from ..harness.provider import (
    AgentRuntimeProvider, CancelOutcome, CloseOutcome, RuntimeEvent, RuntimeEventType,
    RuntimeHealth, RuntimeReceipt, RuntimeSession, RuntimeSessionSpec, RuntimeTurn,
)

AgentRuntime = AgentRuntimeProvider

__all__ = [
    "AgentRuntime", "AgentRuntimeProvider", "CancelOutcome", "CloseOutcome", "RuntimeEvent",
    "RuntimeEventType", "RuntimeHealth", "RuntimeReceipt", "RuntimeSession", "RuntimeSessionSpec",
    "RuntimeTurn",
]
