"""Adapters: one way to put a case to one kind of system and log everything."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from evals.investigation.record import StudyStore, TurnRecord
from evals.investigation.systems import SystemSpec


@dataclass
class AdapterResult:
    status: str
    turns: list[TurnRecord]
    final_text: str = ""
    model: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    toxagent: dict[str, Any] | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None


class Adapter(Protocol):
    def describe(self) -> dict[str, Any]:
        """Versions and configuration, for the study manifest. Never a secret."""

    async def run(
        self, case: dict[str, Any], spec: SystemSpec, *, trial: int,
        snapshot: dict[str, Any] | None, store: StudyStore,
    ) -> AdapterResult: ...
