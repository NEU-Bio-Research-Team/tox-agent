"""P1-1: a usage report is evidence of a state, not an increment.

The audit's Q&A run stored the same cumulative snapshot three times and had no
way to tell. These tests run the captured envelopes through the adapter's
normalizer and assert what survives, and what the summary of it says.
"""
from __future__ import annotations

import pytest

from tests.support.audit_fixtures import DUPLICATE_USAGE_SSE, load
from toxagent.harness.usage_normalizer import (
    RuntimeUsageNormalizer,
    UsageCounts,
    UsageSemantics,
    derive_source_event_id,
    summarize,
)

FIXTURE = load(DUPLICATE_USAGE_SSE)
SESSION = FIXTURE["runtime_session_id"]


def _payloads() -> list[dict]:
    """The adapter's normalized USAGE_REPORTED payloads for the fixture."""
    out: list[dict] = []
    for envelope in FIXTURE["envelopes"]:
        properties = envelope["properties"]
        if envelope["type"] == "message.part.updated":
            part = properties["part"]
            out.append(
                {
                    "tokens": part["tokens"],
                    "source_event_type": "message.part.updated:step-finish",
                    "provider_message_id": part["messageID"],
                    "provider_step_id": part["id"],
                }
            )
        else:
            info = properties["info"]
            out.append(
                {
                    "tokens": info["tokens"],
                    "source_event_type": "message.updated",
                    "provider_message_id": info["id"],
                    "revision": info.get("revision"),
                }
            )
    return out


def _facts(payloads=None):
    normalizer = RuntimeUsageNormalizer(SESSION)
    return [
        fact
        for fact in (normalizer.accept(p) for p in (payloads or _payloads()))
        if fact is not None
    ]


# --- the finding ------------------------------------------------------------


def test_the_audit_stream_yields_one_fact_per_assistant_message() -> None:
    facts = _facts()
    assert len(facts) == FIXTURE["expected_normalized"]["facts"]
    assert {fact.provider_message_id for fact in facts} == {
        "msg_assistant_1",
        "msg_assistant_2",
    }


def test_the_summary_matches_what_the_provider_actually_reported() -> None:
    summary = summarize(_facts())
    expected = FIXTURE["expected_normalized"]["summary"]
    assert summary.input_tokens == expected["input_tokens"]
    assert summary.output_tokens == expected["output_tokens"]
    assert summary.reasoning_tokens == expected["reasoning_tokens"]
    assert summary.cache_read_tokens == expected["cache_read_tokens"]
    assert summary.cache_write_tokens == expected["cache_write_tokens"]
    assert summary.total_tokens == expected["total_tokens"]
    assert summary.aggregation_method == "latest_per_scope"
    assert summary.completeness == "complete"


@pytest.mark.parametrize("repeats", [1, 2, 3])
def test_replaying_the_whole_stream_never_changes_the_total(repeats: int) -> None:
    """Delivering an event once, twice or three times is one fact either way —
    and so is a reconnect that replays the stream from the beginning."""
    normalizer = RuntimeUsageNormalizer(SESSION)
    facts = []
    for _ in range(repeats):
        for payload in _payloads():
            fact = normalizer.accept(payload)
            if fact is not None:
                facts.append(fact)
    assert len(facts) == 2
    assert summarize(facts).input_tokens == 11238


def test_a_zero_only_lifecycle_report_is_not_a_measurement() -> None:
    normalizer = RuntimeUsageNormalizer(SESSION)
    assert (
        normalizer.accept(
            {
                "tokens": {"input": 0, "output": 0, "reasoning": 0, "cache": {"read": 0, "write": 0}},
                "source_event_type": "message.part.updated:step-finish",
                "provider_message_id": "msg_x",
                "provider_step_id": "prt_x",
            }
        )
        is None
    )


def test_a_report_with_no_counts_at_all_is_dropped() -> None:
    normalizer = RuntimeUsageNormalizer(SESSION)
    assert normalizer.accept({"source_event_type": "message.updated"}) is None


# --- ordering ---------------------------------------------------------------


def test_a_late_older_revision_does_not_replace_a_newer_total() -> None:
    normalizer = RuntimeUsageNormalizer(SESSION)
    newer = {
        "tokens": {"input": 900, "output": 90},
        "source_event_type": "message.updated",
        "provider_message_id": "msg_1",
        "revision": 4,
    }
    older = {
        "tokens": {"input": 100, "output": 10},
        "source_event_type": "message.updated",
        "provider_message_id": "msg_1",
        "revision": 2,
    }
    assert normalizer.accept(newer) is not None
    assert normalizer.accept(older) is None


def test_the_summary_picks_the_highest_revision_per_scope() -> None:
    normalizer = RuntimeUsageNormalizer(SESSION)
    facts = [
        normalizer.accept(
            {
                "tokens": {"input": value, "output": 1},
                "source_event_type": "message.updated",
                "provider_message_id": "msg_1",
                "revision": revision,
            }
        )
        for revision, value in ((1, 100), (2, 250), (3, 400))
    ]
    assert all(fact is not None for fact in facts)
    assert summarize(facts).input_tokens == 400


def test_cumulative_totals_never_decrease_across_valid_revisions() -> None:
    normalizer = RuntimeUsageNormalizer(SESSION)
    running: list[int] = []
    for revision, value in enumerate([50, 120, 120, 400, 900], start=1):
        fact = normalizer.accept(
            {
                "tokens": {"input": value, "output": 1},
                "source_event_type": "message.updated",
                "provider_message_id": "msg_1",
                "revision": revision,
            }
        )
        if fact is not None:
            running.append(summarize([fact]).input_tokens or 0)
    assert running == sorted(running)


# --- semantics --------------------------------------------------------------


def test_a_provider_with_no_contract_is_unknown_not_assumed() -> None:
    normalizer = RuntimeUsageNormalizer(SESSION, provider="some-other-runtime")
    fact = normalizer.accept(
        {"tokens": {"input": 10, "output": 2}, "source_event_type": "whatever"}
    )
    assert fact is not None
    assert fact.semantics is UsageSemantics.UNKNOWN
    summary = summarize([fact])
    assert summary.aggregation_method == "none"
    assert summary.completeness == "unknown"
    assert summary.input_tokens is None


def test_deltas_add_and_cumulatives_do_not() -> None:
    from toxagent.harness.usage_normalizer import NormalizedUsage

    deltas = [
        NormalizedUsage(
            source_event_id=f"usg_{i}",
            source_event_type="turn.delta",
            semantics=UsageSemantics.DELTA,
            counts=UsageCounts(input_tokens=100, output_tokens=10),
            provider_message_id=f"msg_{i}",
        )
        for i in range(3)
    ]
    summary = summarize(deltas)
    assert summary.input_tokens == 300
    assert summary.aggregation_method == "sum_of_deltas"


def test_a_mixed_stream_says_it_is_mixed() -> None:
    from toxagent.harness.usage_normalizer import NormalizedUsage

    facts = [
        NormalizedUsage(
            source_event_id="usg_c",
            source_event_type="message.updated",
            semantics=UsageSemantics.CUMULATIVE,
            counts=UsageCounts(input_tokens=100, output_tokens=10),
            provider_message_id="msg_1",
        ),
        NormalizedUsage(
            source_event_id="usg_d",
            source_event_type="turn.delta",
            semantics=UsageSemantics.DELTA,
            counts=UsageCounts(input_tokens=5, output_tokens=1),
            provider_message_id="msg_2",
        ),
    ]
    assert summarize(facts).aggregation_method == "mixed"


def test_an_empty_run_reports_unknown_rather_than_zero() -> None:
    summary = summarize([])
    assert summary.total_tokens is None
    assert summary.completeness == "unknown"
    assert summary.aggregation_method == "none"


# --- identity ---------------------------------------------------------------


def test_a_derived_id_is_a_pure_function_of_the_report() -> None:
    kwargs = dict(
        runtime_session_id=SESSION,
        source_event_type="message.updated",
        provider_message_id="msg_1",
        provider_step_id=None,
        revision=2,
        counts=UsageCounts(input_tokens=10, output_tokens=1),
    )
    assert derive_source_event_id(**kwargs) == derive_source_event_id(**kwargs)


def test_two_steps_of_one_message_are_told_apart_by_what_they_measured() -> None:
    """V1 does not version ``step-finish`` parts, so the counts are the only
    thing distinguishing two real steps of the same message."""
    base = dict(
        runtime_session_id=SESSION,
        source_event_type="message.part.updated:step-finish",
        provider_message_id="msg_1",
        provider_step_id="prt_1",
        revision=None,
    )
    first = derive_source_event_id(**base, counts=UsageCounts(input_tokens=100))
    second = derive_source_event_id(**base, counts=UsageCounts(input_tokens=200))
    assert first != second


def test_a_provider_supplied_id_is_used_verbatim() -> None:
    normalizer = RuntimeUsageNormalizer(SESSION)
    fact = normalizer.accept(
        {
            "tokens": {"input": 10, "output": 1},
            "source_event_id": "provider-native-id",
            "source_event_type": "message.updated",
            "provider_message_id": "msg_1",
        }
    )
    assert fact is not None and fact.source_event_id == "provider-native-id"
    assert normalizer.accept(
        {
            "tokens": {"input": 99, "output": 9},
            "source_event_id": "provider-native-id",
            "source_event_type": "message.updated",
            "provider_message_id": "msg_1",
        }
    ) is None
