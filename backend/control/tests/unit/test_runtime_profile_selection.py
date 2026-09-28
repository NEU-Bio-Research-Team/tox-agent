"""P0-1: the agent a run says it uses is the agent it uses.

The audit's report build recorded a 64-step budget and ran under a 32-step
agent. These tests pin both halves: the registry resolves an intent to the
right named agent and the *real* cap from the shipped profile file, and the
OpenCode adapter sends the agent the spec names rather than its own default.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from toxagent.platform.config import PACKAGE_ROOT, RuntimeSettings
from toxagent.harness.provider import RuntimeSessionSpec
from toxagent.harness.runtime_profiles import (
    QA_PROFILE_PATH,
    REPORT_PROFILE_PATH,
    RuntimeProfileRegistry,
    assert_agent_declared,
    clear_profile_cache,
    registry_from_settings,
)

SHIPPED_PROFILES = PACKAGE_ROOT / "agent_profiles"


def _registry(**overrides) -> RuntimeProfileRegistry:
    kwargs = dict(
        profiles_dir=SHIPPED_PROFILES,
        agent_name="toxagent",
        report_agent_name="toxagent-report",
        max_steps_qa=32,
        max_steps_research=32,
        max_steps_report=64,
    )
    kwargs.update(overrides)
    return RuntimeProfileRegistry(**kwargs)


# --- the registry -----------------------------------------------------------


def test_report_resolves_to_the_report_agent_and_its_own_cap() -> None:
    spec = _registry().resolve("build_report", capability_profile="report_build")
    assert spec.runtime_agent_name == "toxagent-report"
    assert spec.requested_step_cap == 64
    assert spec.effective_step_cap == 64
    assert spec.cap_is_honoured
    assert spec.discrepancy == ""
    assert spec.profile_source == REPORT_PROFILE_PATH


@pytest.mark.parametrize("intent", ["report_qa", "attribution", "evidence_research"])
def test_conversational_intents_resolve_to_the_shared_agent(intent: str) -> None:
    spec = _registry().resolve(intent, capability_profile="report_qa")
    assert spec.runtime_agent_name == "toxagent"
    assert spec.effective_step_cap == 32
    assert spec.profile_source == QA_PROFILE_PATH


def test_the_audit_configuration_is_reported_as_a_discrepancy() -> None:
    """A report asking for 64 steps from the 32-step agent must say so."""
    registry = _registry(select_named_agents=False)
    spec = registry.resolve("build_report", capability_profile="report_build")
    assert spec.runtime_agent_name == "toxagent"
    assert spec.requested_step_cap == 64
    assert spec.effective_step_cap == 32
    assert not spec.cap_is_honoured
    assert "64" in spec.discrepancy and "32" in spec.discrepancy


def test_an_unreadable_profile_is_unknown_not_zero(tmp_path: Path) -> None:
    clear_profile_cache()
    registry = _registry(profiles_dir=tmp_path)
    spec = registry.resolve("build_report", capability_profile="report_build")
    assert spec.effective_step_cap is None
    assert not spec.cap_is_honoured
    assert "could not be read" in spec.discrepancy


def test_a_newer_profile_field_name_is_accepted(tmp_path: Path) -> None:
    """Current OpenCode calls the field ``steps``; V1 calls it ``maxSteps``.

    Reading either keeps the controlled upgrade (WS01's second beat) from
    needing this module changed on the same day the binary is re-pinned.
    """
    clear_profile_cache()
    target = tmp_path / REPORT_PROFILE_PATH
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"agent": {"toxagent-report": {"steps": 48}}}))
    spec = _registry(profiles_dir=tmp_path).resolve(
        "build_report", capability_profile="report_build"
    )
    assert spec.effective_step_cap == 48
    clear_profile_cache()


def test_the_manifest_records_both_numbers() -> None:
    manifest = _registry().resolve("build_report", capability_profile="report_build").to_manifest()
    assert manifest["runtime_agent_name"] == "toxagent-report"
    assert manifest["requested_step_cap"] == 64
    assert manifest["effective_step_cap"] == 64
    assert manifest["step_cap_honoured"] is True


def test_a_deployment_needs_both_agents_when_selection_is_on() -> None:
    assert _registry().required_agent_names() == ("toxagent", "toxagent-report")
    assert _registry(select_named_agents=False).required_agent_names() == ("toxagent",)


def test_build_report_capability_names_the_agent_it_needs() -> None:
    assert _registry().agent_for_capability("build_report") == "toxagent-report"
    assert _registry().agent_for_capability("report_qa") is None


def test_shipped_profiles_declare_the_agents_the_adapter_will_ask_for() -> None:
    """The deployment check: a renamed agent fails here, not at step 32."""
    assert assert_agent_declared(SHIPPED_PROFILES, QA_PROFILE_PATH, "toxagent") == 32
    assert assert_agent_declared(SHIPPED_PROFILES, REPORT_PROFILE_PATH, "toxagent-report") == 64


def test_registry_from_settings_uses_the_configured_names() -> None:
    registry = registry_from_settings(
        RuntimeSettings(), SimpleNamespace(profiles_dir=SHIPPED_PROFILES)
    )
    assert registry.agent_name_for_intent("build_report") == "toxagent-report"
    assert registry.agent_name_for_intent("report_qa") == "toxagent"


# --- the adapter ------------------------------------------------------------


def _spec(**overrides) -> RuntimeSessionSpec:
    kwargs = dict(
        session_id="ses_" + "0" * 32,
        run_id="run_" + "0" * 32,
        provider_id="openai",
        model_id="gpt-5.6-luna",
        profile="report_build",
        system_prompt="",
        system_prompt_hash="",
        tool_schema=(),
        tool_schema_hash="",
        mcp_url="http://127.0.0.1:9/mcp",
        max_steps=64,
        deadline_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    kwargs.update(overrides)
    return RuntimeSessionSpec(**kwargs)


def test_spec_defaults_keep_the_pre_ws01_behaviour() -> None:
    spec = _spec()
    assert spec.runtime_agent_name == ""
    assert spec.effective_max_steps is None
