"""investigator-pilot-v2 and per-phase usage, as pure contracts (#46/#53).

v2 states its accounting, reserves and token-limit enforcement, and two
budgets that differ in anything hash differently -- the label is not the
check. Phase usage never adds two snapshots of one report, never turns an
unknown into zero, never adds currencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from toxagent.application.runs.budget import effective_run_budget
from toxagent.application.runs.phase_usage import phase_usage, summarize_phase
from toxagent.platform.config import PolicySettings, RuntimeSettings

NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)


def test_v2_states_how_it_counts_and_what_it_reserves(monkeypatch):
    monkeypatch.setenv("TOXAGENT_RUN_BUDGET_PRESET", "investigator-pilot-v2")
    budget = effective_run_budget("decision_support", PolicySettings.from_env(),
                                  RuntimeSettings.from_env())
    assert budget.preset == "investigator-pilot-v2"
    assert budget.call_accounting == "total-logical-invocations-v2"
    assert budget.max_tool_calls == 20 and budget.reserved_submission_slots == 6
    assert budget.max_searches == 3 and budget.max_evidence_reads == 8
    assert budget.review_reserve_s == 120 and budget.max_repairs == 1
    assert budget.token_limits["enforcement"] == "observed_only"
    assert budget.deadline_s == 360


def test_v1_is_named_as_the_ceiling_it_always_was(monkeypatch):
    monkeypatch.setenv("TOXAGENT_RUN_BUDGET_PRESET", "investigator-pilot-v1")
    budget = effective_run_budget("decision_support", PolicySettings.from_env(),
                                  RuntimeSettings.from_env())
    assert budget.call_accounting == "action-read-ceiling-v1"


def test_two_budgets_under_one_preset_label_hash_apart(monkeypatch):
    monkeypatch.setenv("TOXAGENT_RUN_BUDGET_PRESET", "investigator-pilot-v1")
    narrow = effective_run_budget("decision_support", PolicySettings.from_env(),
                                  RuntimeSettings.from_env())
    monkeypatch.setenv("TOXAGENT_RUN_DEADLINE_S", "900")
    monkeypatch.setenv("TOXAGENT_MAX_TOOL_CALLS", "50")
    wide = effective_run_budget("decision_support", PolicySettings.from_env(),
                                RuntimeSettings.from_env())
    assert narrow.preset == wide.preset
    assert narrow.sha256 != wide.sha256


@dataclass
class _Event:
    id: str
    runtime_binding_id: str
    semantics: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    total_tokens: int | None = None
    cost_amount: Decimal | None = None
    cost_currency: str | None = None
    provider_message_id: str | None = None
    revision: int | None = None
    model_id: str = "m"


@dataclass
class _Binding:
    id: str
    runtime_manifest: dict


def test_cumulative_reports_count_once_and_deltas_add():
    events = [
        _Event("e1", "b1", "cumulative", 10, 5, provider_message_id="m1", revision=1),
        _Event("e2", "b1", "cumulative", 30, 9, provider_message_id="m1", revision=2),
        _Event("e3", "b1", "delta", 4, 1),
    ]
    summary = summarize_phase(events)
    assert summary["tokens"]["input"] == 34 and summary["tokens"]["output"] == 10
    assert summary["completeness"] == "complete"


def test_unknown_stays_unknown_and_costs_never_add_across_currencies():
    events = [
        _Event("e1", "b1", "delta", 4, 1, cost_amount=Decimal("0.01"), cost_currency="usd"),
        _Event("e2", "b1", "delta", 4, 1, cost_amount=Decimal("2"), cost_currency="eur"),
        _Event("e3", "b1", "delta", 4, 1),
        _Event("e4", "b1", "unknown", 100, 100),
    ]
    summary = summarize_phase(events)
    assert summary["cost"] == {"EUR": "2", "USD": "0.01"}
    assert summary["cost_completeness"] == "partial"
    assert summary["completeness"] == "partial"
    assert summary["tokens"]["input"] == 12
    only_unknown = summarize_phase([_Event("e5", "b1", "unknown", 100, 100)])
    assert only_unknown["tokens"]["input"] is None and only_unknown["completeness"] == "unknown"
    assert summarize_phase([])["completeness"] == "not_measured"


def test_usage_and_calls_are_grouped_by_the_phase_of_their_dispatch():
    bindings = [_Binding("b1", {"phase": "investigator"}), _Binding("b2", {"phase": "review_initial"}),
                _Binding("b3", {"phase": "semantic_repair"})]
    events = [_Event("e1", "b1", "delta", 10, 2), _Event("e2", "b2", "delta", 3, 1)]
    calls = [{"phase": "investigator", "status": "completed"},
             {"phase": "hard_correction", "status": "completed"},
             {"phase": "review_initial", "status": "completed"},
             {"phase": "investigator", "status": "denied"}]
    report = phase_usage(bindings=bindings, usage_events=events, tool_calls=calls)
    phases = report["phases"]
    assert list(phases)[:4] == ["investigator", "hard_correction", "review_initial",
                                "semantic_repair"]
    assert phases["investigator"]["tool_calls"] == {"completed": 1, "denied": 1}
    assert phases["semantic_repair"]["completeness"] == "not_measured"
    assert phases["semantic_repair"]["dispatches"] == 1
    assert report["run_total"]["tokens"]["input"] == 13


def test_a_recovery_continues_the_original_clock(monkeypatch):
    from toxagent.application.runs import scheduler
    from toxagent.domain.ids import MESSAGE, SESSION, new_id
    from toxagent.domain.run import Intent, Lane, Run

    run = Run.create(new_id(SESSION), new_id(MESSAGE), Lane.AGENTIC, Intent.DECISION_SUPPORT,
                     now=datetime.now(timezone.utc), deadline=timedelta(seconds=100))
    assert scheduler._inherited_deadline(run) is None
    monkeypatch.setenv("TOXAGENT_RUN_BUDGET_PRESET", "investigator-pilot-v2")
    inherited = scheduler._inherited_deadline(run)
    assert inherited is not None and inherited.total_seconds() == pytest.approx(100, abs=2)
