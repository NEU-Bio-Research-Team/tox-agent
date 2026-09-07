"""Usage normalization preserves unknowns instead of manufacturing zeroes."""
from __future__ import annotations

from datetime import datetime, timezone

from toxagent.domain.ids import new_id
from toxagent.domain.usage import RuntimeUsageEvent


def _event(payload):
    return RuntimeUsageEvent.from_provider_payload(
        session_id=new_id("ses"),
        run_id=new_id("run"),
        runtime_binding_id=new_id("rtb"),
        provider_id="openai",
        model_id="gpt-test",
        payload=payload,
        reported_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
    )


def test_usage_preserves_reported_zero_and_unreported_fields_as_unknown():
    event = _event({"tokens": {"input": 0, "output": 5, "cache": {"read": 0}}})

    assert event.input_tokens == 0
    assert event.output_tokens == 5
    assert event.cache_read_tokens == 0
    assert event.reasoning_tokens is None
    assert event.cost_amount is None
    assert event.to_dict()["tokens"]["input"] == 0
    assert event.to_dict()["tokens"]["reasoning"] is None


def test_usage_never_assumes_currency_for_a_bare_cost_number():
    event = _event({"tokens": {}, "cost_amount": "0.001"})

    assert event.cost_amount is None
    assert event.cost_currency is None


def test_usage_normalizes_declared_currency_and_decimal_cost():
    event = _event(
        {"tokens": {"input_tokens": 12}, "cost": {"amount": "0.00125000", "currency": "usd"}}
    )

    assert event.input_tokens == 12
    assert event.to_dict()["cost"] == {"amount": "0.00125000", "currency": "USD"}
