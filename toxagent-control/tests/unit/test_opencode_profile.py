"""The captured model surface is deny-all except ToxAgent MCP."""
from __future__ import annotations

import json

from toxagent.config import PROJECT_ROOT


def test_toxagent_opencode_profile_exposes_only_its_own_mcp_namespace():
    profile = json.loads((PROJECT_ROOT / "agent_profiles/opencode/toxagent.json").read_text())
    agent = profile["agent"]["toxagent"]
    assert agent["mode"] == "primary"
    # 32, not the plan's initial 4 (progress log §4.6, then 8, then this):
    # V1's prompt_async has no per-request step field, so this static cap is
    # what actually bounds every turn, and OpenCode counts one tool call as
    # one step. 4 left no room for the one allowed submit_grounded_answer
    # correction attempt once slice-gathering took more than a single step;
    # 8 was still too low once max_tool_calls_per_run (config.py) was raised
    # to 24 for evidence_research's legitimate multi-search workflow — a live
    # sweep (progress log §14.5) hit this cap before ever calling
    # submit_grounded_answer, confirmed by the model's own final text
    # ("Maximum steps reached before submission completed"), captured via
    # harness/gateway.py's diagnostic log. 32 comfortably clears 24 reads
    # plus up to 2 submit attempts.
    assert agent["maxSteps"] == 32
    assert agent["permission"]["*"] == "deny"
    # OpenCode names an MCP tool ``<server>_<tool>``, so the allow rule must be
    # ``toxagent_*``.  An ``mcp_``-prefixed rule matches nothing, leaves the
    # ``*: deny`` rule in force, and the model is handed no tools at all.
    assert agent["permission"]["toxagent_*"] == "allow"
    assert {"read", "edit", "glob", "grep", "list", "bash", "shell", "task", "subagent", "skill", "webfetch", "websearch", "execute"} <= {
        name for name, effect in agent["permission"].items() if effect == "deny"
    }


def test_remote_mcp_template_uses_an_environment_capability_not_a_checked_in_secret():
    template = json.loads(
        (PROJECT_ROOT / "agent_profiles/opencode/toxagent-mcp.remote.json.template").read_text()
    )
    remote = template["mcp"]["toxagent"]
    assert remote["type"] == "remote"
    assert remote["headers"]["Authorization"] == "Bearer {env:TOXAGENT_MCP_CAPABILITY_TOKEN}"
    assert remote["url"] == "{env:TOXAGENT_MCP_RUNTIME_URL}"
