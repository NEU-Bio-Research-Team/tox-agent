"""P1-10: measure the prompt by component, not as one number.

The audit's minimal report loaded 18,149 input tokens on its first turn against
a target of 8,000, and nothing in the system could say where they went. A total
is not actionable; a breakdown that names the largest component is.
"""
from __future__ import annotations

import pytest

from toxagent.harness.context import build_system_prompt, SessionCheckpoint
from toxagent.harness.prompt_budget import (
    CACHEABLE_PREFIX,
    COMPONENTS,
    PROMPT_TARGETS,
    estimate_tokens,
    measure,
    render_tool_schemas,
    split_system_prompt,
)

SCHEMAS = [
    {"name": "get_analysis_slice", "description": "Read one endpoint's numbers."},
    {"name": "submit_grounded_answer", "description": "Submit the final answer."},
]


def _budget(**overrides):
    payload = {
        "profile": "report_qa",
        "policy_prefix": "You are a toxicology screening assistant. " * 40,
        "tool_schemas": SCHEMAS,
        "pinned_facts": "analysis ana_1: CCO",
        "history": "user: what is the hERG score?",
        "user_message": "What is the hERG score?",
    }
    payload.update(overrides)
    return measure(**payload)


# --- the breakdown ----------------------------------------------------------


def test_every_component_is_measured() -> None:
    budget = _budget()
    assert tuple(item.name for item in budget.components) == COMPONENTS
    assert all(item.tokens > 0 for item in budget.components)


def test_the_total_is_the_sum_of_the_parts() -> None:
    budget = _budget()
    assert budget.total_tokens == sum(item.tokens for item in budget.components)


def test_the_budget_names_the_component_to_cut_first() -> None:
    budget = _budget(policy_prefix="tiny", history="a very long history. " * 500)
    assert budget.largest_component.name == "history"


def test_a_profile_over_its_target_says_by_how_much() -> None:
    budget = _budget(policy_prefix="filler. " * 5000)
    assert budget.target_tokens == PROMPT_TARGETS["report_qa"]
    assert budget.over_budget_by > 0
    assert budget.to_manifest()["over_budget_by"] == budget.over_budget_by


def test_a_profile_within_its_target_is_not_over() -> None:
    assert _budget().over_budget_by == 0


def test_a_profile_with_no_agreed_target_reports_none_rather_than_zero() -> None:
    """No target is a fact about the programme, not a licence to spend."""
    budget = _budget(profile="some_new_profile")
    assert budget.target_tokens is None
    assert budget.over_budget_by == 0


# --- cache-ability ----------------------------------------------------------


def test_the_cacheable_prefix_is_the_static_half() -> None:
    budget = _budget()
    cacheable = sum(
        item.tokens for item in budget.components if item.name in CACHEABLE_PREFIX
    )
    assert budget.cacheable_tokens == cacheable
    assert 0 < budget.cacheable_fraction < 1


def test_the_prefix_hash_is_stable_across_turns_of_one_profile() -> None:
    """This is the number to watch: if it moves, no provider cache can hit."""
    first = _budget(user_message="one", history="a", pinned_facts="x")
    second = _budget(user_message="two", history="b", pinned_facts="y")
    assert first.prefix_hash == second.prefix_hash


def test_changing_a_tool_description_moves_the_prefix_hash() -> None:
    changed = [dict(SCHEMAS[0]), {**SCHEMAS[1], "description": "Submit it."}]
    assert _budget().prefix_hash != _budget(tool_schemas=changed).prefix_hash


def test_reordering_the_tool_list_does_not_move_the_hash() -> None:
    """A dict whose iteration order changed between processes is not a changed
    surface, and a hash that moved for that reason sends every cache-hit
    investigation down a false lead."""
    assert _budget().prefix_hash == _budget(tool_schemas=list(reversed(SCHEMAS))).prefix_hash


def test_rendered_schemas_are_deterministic() -> None:
    assert render_tool_schemas(SCHEMAS) == render_tool_schemas(list(reversed(SCHEMAS)))


# --- the estimate -----------------------------------------------------------


def test_the_estimate_is_stable_for_the_same_text() -> None:
    text = "the same prose, twice"
    assert estimate_tokens(text) == estimate_tokens(text)


def test_empty_text_costs_nothing() -> None:
    assert estimate_tokens("") == 0


def test_longer_text_costs_more() -> None:
    assert estimate_tokens("a" * 100) < estimate_tokens("a" * 1000)


def test_the_estimate_is_in_the_right_order_of_magnitude() -> None:
    """Not a provider count and never presented as one — but a number that was
    out by 10x would make every budget meaningless."""
    prose = "The hERG blocker probability is 0.731 for this compound. " * 100
    tokens = estimate_tokens(prose)
    assert 1_000 < tokens < 2_500


# --- measuring a prompt this product actually builds -------------------------


def test_a_real_system_prompt_splits_into_its_three_parts() -> None:
    from toxagent.harness.context import PinnedReference

    prompt = build_system_prompt(
        capability_profile="report_qa",
        checkpoint=SessionCheckpoint(),
        pinned=[PinnedReference(kind="analysis", id="ana_1", summary="CCO")],
        recent_messages=[],
    )
    policy, pinned, history = split_system_prompt(prompt)
    assert "toxicology" in policy.lower() or "ToxAgent" in policy
    assert "ana_1" in pinned
    assert history == ""


def test_a_prompt_with_no_pinned_or_recent_sections_is_all_policy() -> None:
    prompt = build_system_prompt(
        capability_profile="report_qa",
        checkpoint=SessionCheckpoint(),
        pinned=[],
        recent_messages=[],
    )
    policy, pinned, history = split_system_prompt(prompt)
    assert policy == prompt
    assert pinned == "" and history == ""


def test_the_manifest_carries_what_an_investigation_needs() -> None:
    manifest = _budget(context={"model_id": "gpt-5.6-luna"}).to_manifest()
    assert set(manifest) >= {
        "profile",
        "total_tokens",
        "target_tokens",
        "over_budget_by",
        "cacheable_tokens",
        "cacheable_fraction",
        "prefix_sha256",
        "components",
        "largest_component",
        "model_id",
    }
    assert len(manifest["components"]) == len(COMPONENTS)


@pytest.mark.parametrize("profile,target", sorted(PROMPT_TARGETS.items()))
def test_every_declared_target_matches_the_plan(profile: str, target: int) -> None:
    """2,500 for a conversational turn, 8,000 for a report build — the KPI
    table's numbers, in one place rather than in a comment."""
    assert target in (2_500, 8_000)
