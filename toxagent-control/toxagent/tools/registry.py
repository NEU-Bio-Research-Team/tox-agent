"""The typed tool registry — the canonical definition of the tool plane.

Plan section 8.1: this registry is the source of truth, and MCP is a transport
adapter over it. Schema and execution policy therefore cannot disagree, because
there is only one place either is written down.

Visibility is per capability profile. A tool outside the current profile is
absent from ``tools/list`` *and* refused by the runner with the same error a
nonexistent tool produces — so a model cannot map the tool surface by probing
for a different error message (PROD-06).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Final, Mapping

from pydantic import BaseModel

from ..application.policy import Actor
from ..domain.provenance import content_sha256

#: Capability profiles (plan section 8.3). A profile is a closed set: adding a
#: tool to one is a product decision that changes what a model can do, and the
#: eval suite is expected to be re-run when it happens.
PROFILES: Final[dict[str, frozenset[str]]] = {
    "analysis": frozenset(
        {"create_analysis_snapshot", "get_analysis_slice", "submit_grounded_answer"}
    ),
    "report_qa": frozenset(
        {"get_analysis_slice", "get_attribution", "submit_grounded_answer"}
    ),
    "evidence_research": frozenset(
        {
            "get_analysis_slice", "search_toxicology_evidence", "get_evidence_record",
            "submit_grounded_answer",
        }
    ),
    #: Read-only audit. Deliberately without submit_grounded_answer: an auditor
    #: inspects answers, it does not author them.
    "audit_readonly": frozenset({"get_analysis_slice", "get_evidence_record"}),
}


@dataclass(frozen=True)
class ToolContext:
    """Everything a handler is allowed to know about who is calling.

    ``session_id`` and ``run_id`` come from the capability token, never from the
    model's arguments — a tool argument that disagrees with the token loses
    (plan section 8.5).
    """

    session_id: str
    run_id: str
    actor: Actor
    profile: str
    deadline_at: datetime
    language: str = "en"
    call_id: str = ""


@dataclass(frozen=True)
class ToolOutput:
    """A handler's result, before it becomes a transport envelope.

    The three views are separate on purpose: ``canonical`` is what gets stored
    and validated against, ``model_view`` is the bounded projection a model
    sees, and ``ui_view`` is what a human reads. Collapsing them is how a model
    ends up able to cite a number it was never actually shown.
    """

    canonical: dict[str, Any] = field(default_factory=dict)
    model_view: dict[str, Any] = field(default_factory=dict)
    ui_view: dict[str, Any] = field(default_factory=dict)
    observation_ids: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    attachments: tuple[dict[str, Any], ...] = ()


ToolHandler = Callable[[ToolContext, Any], Awaitable[ToolOutput]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    title: str
    description: str
    input_model: type[BaseModel]
    handler: ToolHandler
    profiles: frozenset[str]
    soft_timeout_s: float
    hard_timeout_s: float
    max_retries: int = 0
    #: Whether a repeat with identical arguments may reuse the stored result
    #: rather than doing the work again.
    idempotent: bool = True

    def json_schema(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        return schema

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.json_schema(),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"tool {definition.name!r} is already registered")
        unknown = definition.profiles - set(PROFILES)
        if unknown:
            raise ValueError(f"tool {definition.name!r} names unknown profiles: {sorted(unknown)}")
        for profile in definition.profiles:
            if definition.name not in PROFILES[profile]:
                raise ValueError(
                    f"tool {definition.name!r} claims profile {profile!r}, but the profile does "
                    "not list it; PROFILES is the product decision and wins"
                )
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def visible_for(self, profile: str) -> tuple[ToolDefinition, ...]:
        allowed = PROFILES.get(profile, frozenset())
        return tuple(
            self._tools[name] for name in sorted(allowed) if name in self._tools
        )

    def is_visible(self, name: str, profile: str) -> bool:
        return name in PROFILES.get(profile, frozenset()) and name in self._tools

    def descriptors(self, profile: str) -> list[dict[str, Any]]:
        return [tool.descriptor() for tool in self.visible_for(profile)]

    def schema_hash(self, profile: str | None = None) -> str:
        """Pinned into every runtime binding (PROD-07). If a tool's schema moves,
        the hash moves, and the run audit says which schema produced the answer."""
        descriptors = (
            self.descriptors(profile) if profile
            else [self._tools[n].descriptor() for n in self.names()]
        )
        return content_sha256(descriptors)

    def profile_for_intent(self, intent: str) -> str:
        return {
            "analysis": "analysis",
            "analysis_batch": "analysis",
            "report_qa": "report_qa",
            "attribution": "report_qa",
            "evidence_research": "evidence_research",
        }.get(intent, "report_qa")
