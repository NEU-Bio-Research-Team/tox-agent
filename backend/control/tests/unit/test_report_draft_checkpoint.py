from __future__ import annotations

import pytest

from toxagent.application.submit_report_draft import _apply_draft_patch, _draft_sha256
from toxagent.domain.errors import Conflict
from toxagent.tools.definitions import report as report_tools
from toxagent.tools.registry import ToolRegistry


def test_a_small_patch_updates_a_nested_draft_without_rebuilding_it():
    draft = {
        "title": "Before",
        "sections": [{"body_markdown": "old"}],
        "limitations": [],
    }

    _apply_draft_patch(draft, [
        {"op": "replace", "path": "/sections/0/body_markdown", "value": "new"},
        {"op": "add", "path": "/limitations/-", "value": {"code": "required"}},
    ])

    assert draft["title"] == "Before"
    assert draft["sections"][0]["body_markdown"] == "new"
    assert draft["limitations"] == [{"code": "required"}]


def test_an_unresolvable_patch_is_rejected_instead_of_corrupting_the_checkpoint():
    draft = {"sections": []}

    with pytest.raises(Conflict, match="does not resolve"):
        _apply_draft_patch(
            draft,
            [{"op": "replace", "path": "/sections/4/body_markdown", "value": "new"}],
        )

    assert draft == {"sections": []}


def test_checkpoint_hash_is_canonical_across_mapping_order():
    assert _draft_sha256({"a": 1, "b": [2]}) == _draft_sha256({"b": [2], "a": 1})


def test_the_durable_report_tool_surface_is_registered():
    definitions = {definition.name: definition for definition in report_tools.build(object())}
    registry = ToolRegistry()
    for definition in definitions.values():
        registry.register(definition)

    assert {
        "save_report_draft",
        "check_saved_report_draft",
        "patch_saved_report_draft",
        "submit_saved_report_draft",
    } <= definitions.keys()
    assert definitions["submit_saved_report_draft"].input_model.model_json_schema()[
        "required"
    ] == ["report_build_id", "expected_version"]
    assert registry.is_visible("patch_saved_report_draft", "report_build")
