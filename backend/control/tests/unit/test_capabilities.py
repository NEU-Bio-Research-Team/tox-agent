"""Configured is not available, and mode is not failure.

I01/I02/I05. Three places answered "can this deployment do X" separately and
disagreed: Compose declared a runtime kind the app never builds a provider
for, admission gated `evidence_research` on a provider object existing while
the scheduler was the only thing that knew whether a handler was registered,
and readiness reported the configured kind rather than what was bound.

The concrete failure these tests pin: in the default stack, a request for
report Q&A, attribution or research passed admission, wrote a message and a
run, and only then failed inside the scheduler with "no handler is
registered" — an orphaned run for work the deployment could have refused.
"""
from __future__ import annotations

import pytest

from toxagent.application.capabilities import (
    CONVERSATIONAL,
    CapabilityResolver,
    DeploymentMode,
)
from toxagent.domain.run import Intent


class FakeScheduler:
    def __init__(self, *registered: Intent) -> None:
        self._registered = set(registered)

    def handles(self, intent: Intent) -> bool:
        return intent in self._registered

    def register(self, intent: Intent) -> None:
        self._registered.add(intent)


DETERMINISTIC = (Intent.ANALYSIS, Intent.ANALYSIS_BATCH)


def predictor_only(**kwargs) -> CapabilityResolver:
    return CapabilityResolver(
        scheduler=FakeScheduler(*DETERMINISTIC), runtime_kind="none", **kwargs
    )


def agent_enabled(**kwargs) -> CapabilityResolver:
    gateway = object()
    return CapabilityResolver(
        scheduler=FakeScheduler(*DETERMINISTIC, *CONVERSATIONAL),
        runtime_kind="opencode",
        runtime_gateway_getter=lambda: gateway,
        **kwargs,
    )


# --- mode -------------------------------------------------------------------

def test_a_stack_with_no_runtime_is_predictor_only():
    assert predictor_only().mode is DeploymentMode.PREDICTOR_ONLY


def test_mode_comes_from_what_is_bound_not_from_the_configured_kind():
    """The I01 bug exactly: Compose said `scripted`, nothing was constructed.

    A kind naming a runtime must not by itself make this agent-enabled.
    """
    resolver = CapabilityResolver(
        scheduler=FakeScheduler(*DETERMINISTIC), runtime_kind="scripted"
    )
    assert resolver.mode is DeploymentMode.PREDICTOR_ONLY


def test_a_bound_gateway_with_registered_handlers_is_agent_enabled():
    assert agent_enabled().mode is DeploymentMode.AGENT_ENABLED


def test_a_gateway_bound_after_construction_is_seen():
    """api/app.py builds the resolver before it binds the gateway."""
    scheduler = FakeScheduler(*DETERMINISTIC)
    bound: list[object] = []
    resolver = CapabilityResolver(
        scheduler=scheduler,
        runtime_kind="opencode",
        runtime_gateway_getter=lambda: bound[0] if bound else None,
    )
    assert resolver.mode is DeploymentMode.PREDICTOR_ONLY
    for intent in CONVERSATIONAL:
        scheduler.register(intent)
    bound.append(object())
    assert resolver.mode is DeploymentMode.AGENT_ENABLED


# --- per-intent availability ------------------------------------------------

def test_prediction_is_available_in_every_mode():
    for resolver in (predictor_only(), agent_enabled()):
        assert resolver.available(Intent.ANALYSIS)
        assert resolver.available(Intent.ANALYSIS_BATCH)


@pytest.mark.parametrize("intent", sorted(CONVERSATIONAL, key=lambda i: i.value))
def test_a_conversational_intent_is_unavailable_without_a_runtime(intent):
    capability = predictor_only().intent(intent)
    assert capability.available is False
    assert capability.reason, "an unavailable capability must say why"
    # Written for whoever reads it, not as a registry miss.
    assert "runtime" in capability.reason


def test_report_qa_and_attribution_were_the_ungated_pair():
    """These two reached the scheduler with nothing registered (I02)."""
    resolver = predictor_only()
    assert not resolver.available(Intent.REPORT_QA)
    assert not resolver.available(Intent.ATTRIBUTION)


def test_research_needs_a_provider_as_well_as_a_runtime():
    with_runtime_no_provider = agent_enabled(research_provider=None)
    capability = with_runtime_no_provider.intent(Intent.EVIDENCE_RESEARCH)
    assert capability.available is False
    assert "provider" in capability.reason

    with_both = agent_enabled(research_provider=object())
    assert with_both.available(Intent.EVIDENCE_RESEARCH)


def test_a_research_provider_alone_does_not_make_research_available():
    """The precise I02 inversion: a provider existing was taken as capability."""
    resolver = predictor_only(research_provider=object())
    assert resolver.available(Intent.EVIDENCE_RESEARCH) is False


def test_structure_recognition_follows_the_ocr_service():
    assert predictor_only(ocr_client=None).available(Intent.STRUCTURE_RECOGNITION) is False
    with_ocr = CapabilityResolver(
        scheduler=FakeScheduler(*DETERMINISTIC, Intent.STRUCTURE_RECOGNITION),
        runtime_kind="none",
        ocr_client=object(),
    )
    assert with_ocr.available(Intent.STRUCTURE_RECOGNITION)


# --- the reported shape -----------------------------------------------------

def test_every_requestable_intent_is_reported_with_a_timestamp():
    snapshot = predictor_only().snapshot()
    for name, capability in snapshot.items():
        payload = capability.to_dict()
        assert set(payload) >= {"configured", "available", "checked_at"}
        if not capability.available:
            assert payload["reason"], name


def test_configured_and_available_are_reported_separately():
    """A stack that declared a runtime but bound none must show both facts."""
    resolver = CapabilityResolver(
        scheduler=FakeScheduler(*DETERMINISTIC, *CONVERSATIONAL),
        runtime_kind="opencode",
        runtime_gateway_getter=lambda: None,
    )
    capability = resolver.intent(Intent.REPORT_QA)
    assert capability.configured is True
    assert capability.available is False


def test_router_outcomes_are_not_capabilities():
    snapshot = predictor_only().snapshot()
    assert "clarification_required" not in snapshot
    assert "out_of_scope" not in snapshot
