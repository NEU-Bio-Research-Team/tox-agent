"""The OpenCode config a runtime host loads declares every agent the product dispatches.

Found live on 2026-09-13: `build_report` was unavailable on a local agent stack
because the launcher points `OPENCODE_CONFIG` at `opencode/toxagent.json`, and
that file declared only `toxagent`. `toxagent-report` existed in
`report_build/profile.json` — which the control plane reads for the step cap and
no runtime ever loads. The capability resolver was right to refuse; the deployed
config was wrong, and no test compared the two files.
"""
from __future__ import annotations

import json

from toxagent.config import PACKAGE_ROOT
from toxagent.harness.runtime_profiles import QA_PROFILE_PATH, REPORT_PROFILE_PATH

PROFILES = PACKAGE_ROOT / "agent_profiles"


def _agents(relative: str) -> dict:
    return json.loads((PROFILES / relative).read_text())["agent"]


def test_the_loaded_config_declares_the_report_agent():
    assert set(_agents(QA_PROFILE_PATH)) >= {"toxagent", "toxagent-report"}


def test_the_report_agent_the_runtime_loads_matches_the_one_the_manifest_reads():
    """Same cap and same permissions, or the manifest describes an agent that
    is not the one that runs (P0-1 in a new place)."""
    loaded = _agents(QA_PROFILE_PATH)["toxagent-report"]
    declared = _agents(REPORT_PROFILE_PATH)["toxagent-report"]
    assert loaded["maxSteps"] == declared["maxSteps"]
    assert loaded["permission"] == declared["permission"]
    assert loaded["mode"] == declared["mode"]


def test_both_agents_are_deny_all_but_their_own_mcp_namespace():
    for name, agent in _agents(QA_PROFILE_PATH).items():
        permission = agent["permission"]
        allowed = {key for key, effect in permission.items() if effect == "allow"}
        assert permission["*"] == "deny", name
        assert allowed == {"toxagent_*"}, name
