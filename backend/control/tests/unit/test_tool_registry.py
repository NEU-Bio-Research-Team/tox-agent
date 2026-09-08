"""Registry and profile visibility (plan sections 8.1, 8.3)."""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from toxagent.tools.registry import PROFILES, ToolDefinition, ToolRegistry


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


def test_every_profile_is_a_small_closed_set():
    """Plan section 21: a large tool roster costs money and misroutes."""
    for name, tools in PROFILES.items():
        assert 2 <= len(tools) <= 6, f"{name} has {len(tools)} tools"


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
