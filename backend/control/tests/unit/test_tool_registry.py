"""Registry and profile visibility (plan sections 8.1, 8.3)."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from toxagent.tools.registry import FLAG_GATED_TOOLS, PROFILES, ToolDefinition, ToolRegistry


class Args(BaseModel):
    x: int


async def handler(context, payload):  # pragma: no cover - never called here
    raise AssertionError("not invoked")


def definition(name: str, profiles: frozenset[str]) -> ToolDefinition:
    return ToolDefinition(
        name=name, title=name, description="", input_model=Args, handler=handler,
        profiles=profiles, soft_timeout_s=1.0, hard_timeout_s=2.0,
    )


def test_a_tool_cannot_claim_a_profile_that_does_not_list_it():
    registry = ToolRegistry()
    # get_attribution is not in the audit_readonly profile, so claiming it is a
    # registration error rather than a quiet widening of what an auditor can do.
    with pytest.raises(ValueError, match="PROFILES is the product decision"):
        registry.register(definition("get_attribution", frozenset({"audit_readonly"})))


def test_an_unknown_profile_is_refused():
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="unknown profiles"):
        registry.register(definition("get_analysis_slice", frozenset({"root"})))


def test_registering_a_name_twice_is_an_error():
    registry = ToolRegistry()
    registry.register(definition("get_analysis_slice", frozenset({"analysis"})))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(definition("get_analysis_slice", frozenset({"analysis"})))


def test_the_audit_profile_cannot_author_an_answer():
    """Plan section 8.3: submit_grounded_answer is for the product agent only."""
    assert "submit_grounded_answer" not in PROFILES["audit_readonly"]


#: The one profile that is allowed to be large, and exactly how large. A report
#: build is the single workflow that has to assemble substance, predictor,
#: explanation and evidence facts within one run, so report spec section 8
#: enumerates a wider roster on purpose. Pinning it by name here keeps the
#: "small closed set" guard meaningful for every other profile while making any
#: drift in this one a failing test rather than a quiet expansion.
REPORT_BUILD_TOOLS = frozenset(
    {
        "get_report_context",
        "get_analysis_bundle",
        "get_analysis_slice",
        "resolve_compound_record",
        "get_or_create_explanation",
        "get_explanation_package",
        "search_toxicology_evidence",
        "get_evidence_record",
        "save_report_draft",
        "check_saved_report_draft",
        "patch_saved_report_draft",
        "submit_saved_report_draft",
        "check_report_draft",
        "submit_report_draft",
    }
)


def test_every_conversational_profile_is_a_small_closed_set():
    """Plan section 21: a large tool roster costs money and misroutes."""
    for name, tools in PROFILES.items():
        # None is conversational: report_build is the model-driven builder's
        # enumerated roster, and report_synthesis and claim_review (W9-12) are
        # one submission boundary each.
        if name in ("report_build", "report_synthesis", "claim_review"):
            continue
        # decision_support is deliberately the adaptive superset of
        # report_qa + evidence_research (ADR 0010, ADS plan section 7.2) plus
        # the read-your-own-artifacts tools of W2-03/04 — its ceiling is wider
        # on purpose, not an oversight this guardrail should catch.
        ceiling = 10 if name == "decision_support" else 6
        default = tools - set(FLAG_GATED_TOOLS)
        assert 2 <= len(default) <= ceiling, f"{name} has {len(default)} default tools"
        # Every flag on at once (ADR 0012: the case and skill tools are cheap
        # reads/writes of product state, not new capabilities to route among)
        # still has a ceiling, so a flag cannot become a way around this one.
        assert len(tools) <= ceiling + 5, f"{name} has {len(tools)} tools with every flag on"


def test_the_orchestrated_synthesis_turn_sees_exactly_one_tool():
    """PR-12: by dispatch time the server has done every read a model could
    ask for, so the only thing left to expose is the submission."""
    assert PROFILES["report_synthesis"] == frozenset({"submit_report_synthesis"})


def test_the_report_build_roster_is_exactly_what_the_spec_enumerates():
    assert PROFILES["report_build"] == REPORT_BUILD_TOOLS


def test_a_report_build_cannot_author_a_conversational_answer():
    """A report is not an answer: two ways to finish a run would mean two
    validators to satisfy and two things a transcript could call the result."""
    assert "submit_grounded_answer" not in PROFILES["report_build"]


def test_visibility_follows_the_profile(registry_with_two_tools):
    registry = registry_with_two_tools
    assert [t.name for t in registry.visible_for("analysis")] == ["get_analysis_slice"]
    assert registry.is_visible("get_analysis_slice", "analysis")
    assert not registry.is_visible("get_attribution", "analysis")
    assert registry.is_visible("get_attribution", "report_qa")


def test_the_schema_hash_changes_when_a_schema_changes(registry_with_two_tools):
    before = registry_with_two_tools.schema_hash("report_qa")

    class Wider(BaseModel):
        x: int
        y: str = ""

    other = ToolRegistry()
    other.register(
        ToolDefinition(
            name="get_attribution", title="t", description="", input_model=Wider,
            handler=handler, profiles=frozenset({"report_qa"}),
            soft_timeout_s=1.0, hard_timeout_s=2.0,
        )
    )
    assert other.schema_hash("report_qa") != before


def test_the_schema_hash_is_per_profile(registry_with_two_tools):
    assert registry_with_two_tools.schema_hash("analysis") != registry_with_two_tools.schema_hash(
        "report_qa"
    )


@pytest.fixture
def registry_with_two_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(definition("get_analysis_slice", frozenset({"analysis", "report_qa"})))
    registry.register(definition("get_attribution", frozenset({"report_qa"})))
    return registry
