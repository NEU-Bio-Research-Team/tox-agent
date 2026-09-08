"""The live-surface check catches what reading the checked-in profile cannot.

Progress log §4.2: ``serve --pure`` alone still let the machine's
``~/.opencode`` leak ``read: allow`` and extra MCP namespaces into the resolved
agent. ``evaluate_surface`` is fed the shape ``GET /agent`` actually returns and
must flag exactly that — and must not flag the shape a properly isolated server
returns.
"""
from __future__ import annotations

from assert_opencode_surface import evaluate_surface

# --- the flat-dict shape the checked-in profile is authored in --------------

CLEAN_AGENT = {
    "name": "toxagent",
    "mode": "primary",
    "permission": {
        "*": "deny",
        "read": "deny",
        "edit": "deny",
        "bash": "deny",
        "webfetch": "deny",
        "websearch": "deny",
        "task": "deny",
        "toxagent_*": "allow",
    },
}


def test_a_deny_all_agent_with_only_its_own_mcp_allow_has_no_problems():
    assert evaluate_surface([CLEAN_AGENT], "toxagent") == []


def test_the_mcp_prefixed_naming_is_also_accepted():
    agent = {**CLEAN_AGENT, "permission": {"*": "deny", "mcp_toxagent_get_analysis_slice": "allow"}}
    assert evaluate_surface([agent], "toxagent") == []


def test_a_leaked_read_allow_is_flagged():
    agent = {**CLEAN_AGENT, "permission": {**CLEAN_AGENT["permission"], "read": "allow"}}
    problems = evaluate_surface([agent], "toxagent")
    assert any("read" in p and "allow" in p for p in problems)


def test_a_leaked_foreign_mcp_namespace_is_flagged():
    agent = {
        **CLEAN_AGENT,
        "permission": {**CLEAN_AGENT["permission"], "codegraph_search": "allow", "officecli": "allow"},
    }
    problems = evaluate_surface([agent], "toxagent")
    assert any("codegraph_search" in p for p in problems)
    assert any("officecli" in p for p in problems)


def test_a_missing_default_deny_is_flagged():
    agent = {**CLEAN_AGENT, "permission": {"toxagent_*": "allow"}}
    problems = evaluate_surface([agent], "toxagent")
    assert any('"*"' in p for p in problems)


def test_no_allow_rule_for_own_mcp_means_the_model_gets_no_tools():
    agent = {**CLEAN_AGENT, "permission": {"*": "deny", "read": "deny"}}
    problems = evaluate_surface([agent], "toxagent")
    assert any("no allow rule" in p for p in problems)


def test_a_missing_agent_is_a_problem_not_a_crash():
    assert evaluate_surface([{"name": "build"}], "toxagent") == [
        "agent 'toxagent' is not present in GET /agent"
    ]


def test_external_directory_allow_at_the_wildcard_pattern_is_flagged():
    agent = {
        **CLEAN_AGENT,
        "permission": {**CLEAN_AGENT["permission"], "external_directory": "allow"},
    }
    problems = evaluate_surface([agent], "toxagent")
    assert any("external_directory" in p for p in problems)


# --- the real ordered-rule-list shape GET /agent returns (OpenCode 1.17.11) -

#: Captured live from an isolated ``opencode serve --pure`` (progress log
#: §3.6): OpenCode's own agent-mode defaults first (``* allow``, ``read
#: allow``, narrow ``external_directory allow`` grants for its own
#: tool-output cache, …), then the checked-in profile's rules appended.
LIVE_TOXAGENT_PERMISSION_LIST = [
    {"permission": "*", "pattern": "*", "action": "allow"},
    {"permission": "doom_loop", "pattern": "*", "action": "ask"},
    {"permission": "external_directory", "pattern": "*", "action": "ask"},
    {"permission": "external_directory", "pattern": "/home/u/.local/share/opencode/tool-output/*", "action": "allow"},
    {"permission": "external_directory", "pattern": "/tmp/opencode/*", "action": "allow"},
    {"permission": "question", "pattern": "*", "action": "deny"},
    {"permission": "plan_enter", "pattern": "*", "action": "deny"},
    {"permission": "plan_exit", "pattern": "*", "action": "deny"},
    {"permission": "read", "pattern": "*", "action": "allow"},
    {"permission": "read", "pattern": "*.env", "action": "ask"},
    {"permission": "read", "pattern": "*.env.*", "action": "ask"},
    {"permission": "read", "pattern": "*.env.example", "action": "allow"},
    {"permission": "*", "pattern": "*", "action": "deny"},
    {"permission": "read", "pattern": "*", "action": "deny"},
    {"permission": "edit", "pattern": "*", "action": "deny"},
    {"permission": "glob", "pattern": "*", "action": "deny"},
    {"permission": "grep", "pattern": "*", "action": "deny"},
    {"permission": "list", "pattern": "*", "action": "deny"},
    {"permission": "bash", "pattern": "*", "action": "deny"},
    {"permission": "shell", "pattern": "*", "action": "deny"},
    {"permission": "task", "pattern": "*", "action": "deny"},
    {"permission": "subagent", "pattern": "*", "action": "deny"},
    {"permission": "skill", "pattern": "*", "action": "deny"},
    {"permission": "webfetch", "pattern": "*", "action": "deny"},
    {"permission": "websearch", "pattern": "*", "action": "deny"},
    {"permission": "execute", "pattern": "*", "action": "deny"},
    {"permission": "toxagent_*", "pattern": "*", "action": "allow"},
    {"permission": "external_directory", "pattern": "/home/u/.local/share/opencode/tool-output/*", "action": "allow"},
]

LIVE_TOXAGENT_AGENT = {
    "name": "toxagent", "mode": "primary", "native": False,
    "permission": LIVE_TOXAGENT_PERMISSION_LIST,
}


def test_the_live_isolated_capture_has_no_problems():
    """The exact payload GET /agent returned once run_local_phase3.sh's env -i
    + isolated HOME/XDG isolation was in place (progress §3.6)."""
    assert evaluate_surface([LIVE_TOXAGENT_AGENT], "toxagent") == []


def test_narrow_external_directory_grants_at_non_star_patterns_are_ignored():
    """OpenCode's own tool-output/tmp carve-outs use specific paths, never the
    wildcard pattern; only a wildcard allow would be a real filesystem escape."""
    problems = evaluate_surface([LIVE_TOXAGENT_AGENT], "toxagent")
    assert not any("external_directory" in p for p in problems)


def test_a_leaked_read_allow_as_the_last_list_entry_is_flagged():
    """If a leaked global config re-appended read: allow *after* the profile's
    own read: deny, the running server would resolve read to allow — and this
    must catch it even though the profile file itself still says deny."""
    leaked = [*LIVE_TOXAGENT_PERMISSION_LIST, {"permission": "read", "pattern": "*", "action": "allow"}]
    agent = {**LIVE_TOXAGENT_AGENT, "permission": leaked}
    problems = evaluate_surface([agent], "toxagent")
    assert any("read" in p and "allow" in p for p in problems)


def test_a_leaked_foreign_mcp_in_the_live_shape_is_flagged():
    leaked = [
        *LIVE_TOXAGENT_PERMISSION_LIST,
        {"permission": "codegraph_search", "pattern": "*", "action": "allow"},
    ]
    agent = {**LIVE_TOXAGENT_AGENT, "permission": leaked}
    problems = evaluate_surface([agent], "toxagent")
    assert any("codegraph_search" in p for p in problems)


def test_ui_interaction_permissions_resolving_allow_are_not_flagged():
    """question/plan_enter/plan_exit/doom_loop gate a chat-mode interaction,
    not tool/filesystem/network access; the native `build` agent legitimately
    resolves some of these to allow and that is not a leaked capability."""
    permission = [
        *LIVE_TOXAGENT_PERMISSION_LIST,
        {"permission": "question", "pattern": "*", "action": "allow"},
        {"permission": "plan_enter", "pattern": "*", "action": "allow"},
    ]
    agent = {**LIVE_TOXAGENT_AGENT, "permission": permission}
    assert evaluate_surface([agent], "toxagent") == []
