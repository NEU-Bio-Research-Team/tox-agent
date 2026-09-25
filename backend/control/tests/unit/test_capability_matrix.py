"""The checked-in capability matrix matches the code (RETHINK §6 P0).

When this fails, what a default deployment can do changed — a model admitted or
blocked, a flag added, a tool moved between profiles, an instruction file
grown. Regenerate with ``python -m evals.capability_matrix --write`` and review
the diff: it is the capability change, stated once.
"""
from __future__ import annotations

from evals.capability_matrix import (
    JSON_PATH, MARKDOWN_PATH, build_matrix, render_json, render_markdown,
)
from toxagent import flags as rollout
from toxagent.tools.registry import FLAG_GATED_TOOLS, PROFILES


def test_the_checked_in_matrix_is_current():
    matrix = build_matrix()
    assert JSON_PATH.read_text() == render_json(matrix), (
        "docs/capability-matrix.json is stale; run `python -m evals.capability_matrix --write`"
    )
    assert MARKDOWN_PATH.read_text() == render_markdown(matrix), (
        "docs/CAPABILITY_MATRIX.md is stale; run `python -m evals.capability_matrix --write`"
    )


def test_a_blocked_model_says_why_and_serves_nothing():
    matrix = build_matrix()
    for model in matrix["predictor_models"]:
        if model["status"] == "blocked":
            assert model["blocked_reason"]
            for endpoint in model["endpoints"]:
                entry = matrix["endpoints"][endpoint]
                assert entry["status"] == "blocked" or entry["model_id"] != model["model_id"]


def test_the_served_hERG_model_is_chemberta_not_a_gnn():
    """RETHINK §2: GNN/GNNExplainer is research history, not the served capability."""
    endpoints = build_matrix()["endpoints"]
    assert endpoints["herg"] == {"status": "served", "model_id": "herg-tox21-chemberta-v1"}
    assert endpoints["clintox"]["status"] == "blocked"


def test_every_flag_gated_tool_names_a_real_flag_and_a_real_profile():
    names = {flag.name for flag in rollout.FLAGS}
    for tool, flag_name in FLAG_GATED_TOOLS.items():
        assert flag_name in names, (tool, flag_name)
        assert any(tool in tools for tools in PROFILES.values()), tool


def test_a_flag_gated_tool_is_never_reported_as_a_default_capability():
    for profile in build_matrix()["tool_profiles"].values():
        assert not set(profile["default"]) & set(FLAG_GATED_TOOLS)
