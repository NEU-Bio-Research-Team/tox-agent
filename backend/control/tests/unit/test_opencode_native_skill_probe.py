"""The W9-14 probe's workspace is isolated the way RETHINK §4.9 asks."""
from __future__ import annotations

import json

from evals.experiments.opencode_native_skills import AGENT, _env, _workspace


def test_only_approved_decision_support_skills_are_offered(tmp_path):
    project, config_path, pins = _workspace(tmp_path)
    shipped = sorted(p.name for p in (project / ".opencode" / "skills").iterdir())
    assert shipped == sorted(pin["skill_id"] for pin in pins)
    assert "compose-scientific-report" not in shipped  # report skills stay out
    assert not list(project.rglob("skill.manifest.json"))
    config = json.loads(config_path.read_text())
    permission = config["agent"][AGENT]["permission"]
    assert permission["*"] == "deny" and permission["read"] == "deny"
    assert permission["skill"]["*"] == "deny"  # the built-in customize-opencode too
    assert {k for k, v in permission["skill"].items() if v == "allow"} == set(shipped)


def test_the_denied_arm_turns_skill_off(tmp_path):
    _, config_path, _ = _workspace(tmp_path, "denied")
    assert json.loads(config_path.read_text())["agent"][AGENT]["permission"]["skill"] == "deny"


def test_the_server_sees_no_home_skills(tmp_path, monkeypatch):
    import evals.experiments.opencode_native_skills as probe

    auth = tmp_path / "repo" / ".data" / "opencode-auth" / "data" / "opencode"
    auth.mkdir(parents=True)
    (auth / "auth.json").write_text("{}")
    monkeypatch.setattr(probe, "REPO", tmp_path / "repo")
    env = _env(tmp_path, tmp_path / "opencode.json")
    assert env["HOME"] == str(tmp_path / "home")
    assert env["OPENCODE_DISABLE_EXTERNAL_SKILLS"] == "1"
    assert env["OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"] == "1"
