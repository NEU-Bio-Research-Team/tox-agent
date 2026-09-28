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
from typing import Any, Awaitable, Callable, Final

from pydantic import BaseModel

from ..application.tool_context import ToolContext
from ..domain.provenance import content_sha256
from ..application.tool_profiles import manifest as load_profile_manifest

#: Capability profiles (plan section 8.3). A profile is a closed set: adding a
#: tool to one is a product decision that changes what a model can do, and the
#: eval suite is expected to be re-run when it happens. The decision is written
#: down in ``agent_profiles/tool_profiles.json`` (W9-09), validated at import
#: by ``profile_manifest``; this module only exposes it under the names every
#: consumer already reads.
PROFILE_MANIFEST: Final = load_profile_manifest()
PROFILES: Final[dict[str, frozenset[str]]] = dict(PROFILE_MANIFEST.profiles)

#: Tools a profile lists but a deployment registers only while a rollout flag
#: is on (``tools/bootstrap.py`` applies it; ``evals/capability_matrix.py``
#: reports it). Declared in the same manifest, so the bootstrap and the
#: published capability matrix cannot disagree about which flag gates which
#: tool. With the flag off the tool is absent from ``tools/list`` and from the
#: profile's schema hash.
FLAG_GATED_TOOLS: Final[dict[str, str]] = dict(PROFILE_MANIFEST.flag_gated_tools)


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
    #: A model-facing cost signal (ADS plan W2-06/07), distinct from
    #: soft/hard_timeout_s (an execution ceiling, not a reuse hint):
    #: "cheap" — a bounded read of already-persisted state; "moderate" — calls
    #: an external provider but does not mint a new stored artifact each time;
    #: "expensive" — computes and persists something new (e.g. a fresh
    #: attribution), so a repeat with the same arguments is real work reused,
    #: not free. Surfaced in ``descriptor()`` so the model can see it without
    #: it being restated in every tool's description string.
    cost_class: str = "cheap"

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
            "costClass": self.cost_class,
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
            # Historical only (ADR 0010): the router no longer produces these
            # three; kept so a stray legacy value still resolves to a real
            # profile instead of a KeyError.
            "report_qa": "report_qa",
            "attribution": "report_qa",
            "evidence_research": "evidence_research",
            "decision_support": "decision_support",
            "build_report": "report_build",
        }.get(intent, "decision_support")
