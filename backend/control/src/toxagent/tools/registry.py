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
        {
            "create_analysis_snapshot", "get_analysis_slice",
            "submit_grounded_answer",
        }
    ),
    "report_qa": frozenset(
        {
            "get_analysis_slice",
            "get_attribution", "submit_grounded_answer",
        }
    ),
    "evidence_research": frozenset(
        {
            "get_analysis_slice",
            "search_toxicology_evidence", "get_evidence_record",
            "submit_grounded_answer",
        }
    ),
    #: Adaptive decision support (ADS plan section 7.2, ADR 0010). Superset of
    #: report_qa + evidence_research's read/research surface, plus
    #: get_artifact_inventory/get_report_summary (W2-03/04) so a follow-up
    #: turn can see what a prior report already established instead of
    #: guessing from prose: the model picks which of these to call and in
    #: what order within the run's budget, rather than the profile being
    #: pre-selected by a keyword. Deliberately still closed — no
    #: shell/filesystem/raw web, same as every other profile — and still
    #: without the report_build-only draft/submit tooling.
    "decision_support": frozenset(
        {
            "get_artifact_inventory", "get_report_summary",
            "get_analysis_slice", "get_analysis_bundle",
            "get_explanation_slice",
            "get_attribution",
            "search_toxicology_evidence", "get_evidence_record",
            "submit_grounded_answer",
            # TAB-Suite Wave 2: listed here, registered only behind the
            # decision_state_plan_tool flag (tools/bootstrap.py), so it is
            # absent from tools/list until then.
            "record_decision_plan",
            # ADR 0012: the cross-turn case, behind scientific_case_v1, and
            # scientific skills read on demand, behind scientific_skills_v1.
            "get_scientific_case", "update_scientific_case",
            "read_scientific_skill", "read_skill_reference",
        }
    ),
    #: Read-only audit. Deliberately without submit_grounded_answer: an auditor
    #: inspects answers, it does not author them.
    "audit_readonly": frozenset(
        {"get_analysis_slice", "get_evidence_record", "get_explanation_slice"}
    ),
    #: The report builder (report spec section 8). Deliberately *not* a
    #: superset of report_qa: it has no ``submit_grounded_answer``, because a
    #: report is not an answer and a run that could emit either would have two
    #: ways to finish and two validators to satisfy. Note also what is absent
    #: on the figure side — the runtime can ask for an explanation and receive
    #: refs, but never reaches attachment storage or the renderer directly
    #: (spec section 8: "Do not expose both low-level figure rendering and
    #: attachment storage to the agent").
    "report_build": frozenset(
        {
            "get_report_context",
            "get_analysis_bundle", "get_analysis_slice",
            "resolve_compound_record",
            "get_or_create_explanation", "get_explanation_package",
            "search_toxicology_evidence", "get_evidence_record",
            # The dry run sits beside the submission deliberately: it runs the
            # same validator and stores nothing, so a model can find its own
            # bookkeeping slips without spending the build's one correction
            # attempt on the discovery.
            "save_report_draft", "check_saved_report_draft",
            "patch_saved_report_draft", "submit_saved_report_draft",
            # Backward-compatible stateless path. New profiles instruct the
            # runtime to use the durable flow above.
            "check_report_draft", "submit_report_draft",
        }
    ),
    #: The one LLM boundary of an orchestrated report build (WS05 5B / PR-12).
    #: A single tool, on purpose: by the time this profile is dispatched the
    #: server has already resolved the substance, projected the predictions,
    #: produced the explanations and run the search. A read tool here would be
    #: an invitation to redo that work, and a draft tool would be a second way
    #: to finish.
    "report_synthesis": frozenset({"submit_report_synthesis"}),
}


#: Tools a profile lists but a deployment registers only while a rollout flag
#: is on (``tools/bootstrap.py`` applies it; ``evals/capability_matrix.py``
#: reports it). One table, so the bootstrap and the published capability
#: matrix cannot disagree about which flag gates which tool. With the flag off
#: the tool is absent from ``tools/list`` and from the profile's schema hash.
FLAG_GATED_TOOLS: Final[dict[str, str]] = {
    "record_decision_plan": "decision_state_plan_tool",
    "get_scientific_case": "scientific_case_v1",
    "update_scientific_case": "scientific_case_v1",
    "read_scientific_skill": "scientific_skills_v1",
    "read_skill_reference": "scientific_skills_v1",
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
    #: Immutable run intent carried by the signed capability token. It is
    #: presentation context only; tool authorization remains ``profile``.
    intent: str = ""
    call_id: str = ""
    #: The run's resolved predictor binding: endpoint -> admitted model id.
    #: Injected by the server from the run configuration, exactly like
    #: session_id and run_id, and for the same reason — a tool argument that
    #: let a model choose its own provider would put scientific model
    #: selection in the hands of the thing being explained (I10). No tool
    #: input schema exposes a model field; this is the only way one arrives.
    model_selection: Mapping[str, str] | None = None


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
