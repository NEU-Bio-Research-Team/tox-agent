"""Immutable, normalized runtime usage reports (remaining-plan W2-13/14).

A provider report is evidence that it reported *something*, not proof that a
missing field means zero.  Each report therefore remains an event row and every
unreported quantity is ``None``.  We intentionally do not aggregate here:
OpenCode can emit per-step usage while another provider may emit a cumulative
turn total, and summing both without an explicit provider contract would make a
cost dashboard look precise while being wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .ids import RUN, RUNTIME_BINDING, RUNTIME_USAGE, SESSION, new_id, require_id


def _count(value: Any) -> int | None:
    """Return a non-negative integer; malformed/absent is unknown, not zero."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _first_count(values: tuple[Any, ...]) -> int | None:
    for value in values:
        normalized = _count(value)
        if normalized is not None:
            return normalized
    return None


def _amount(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        normalized = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return normalized if normalized >= 0 else None


@dataclass(frozen=True, slots=True)
class RuntimeUsageEvent:
    id: str
    session_id: str
    run_id: str
    runtime_binding_id: str
    provider_id: str
    model_id: str
    reported_at: datetime
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    total_tokens: int | None = None
    cost_amount: Decimal | None = None
    cost_currency: str | None = None

    def __post_init__(self) -> None:
        require_id(self.id, RUNTIME_USAGE, field="usage.id")
        require_id(self.session_id, SESSION, field="usage.session_id")
        require_id(self.run_id, RUN, field="usage.run_id")
        require_id(self.runtime_binding_id, RUNTIME_BINDING, field="usage.runtime_binding_id")
        for name in (
            "input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens",
            "cache_write_tokens", "total_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"usage.{name} must be non-negative when reported")
        if self.cost_amount is not None and self.cost_amount < 0:
            raise ValueError("usage.cost_amount must be non-negative when reported")
        if self.cost_amount is not None and not self.cost_currency:
            raise ValueError("usage.cost_currency is required when cost_amount is reported")

    @classmethod
    def from_provider_payload(
        cls,
        *,
        session_id: str,
        run_id: str,
        runtime_binding_id: str,
        provider_id: str,
        model_id: str,
        payload: Mapping[str, Any],
        reported_at: datetime,
    ) -> "RuntimeUsageEvent":
        tokens = payload.get("tokens")
        tokens = tokens if isinstance(tokens, Mapping) else {}
        cache = tokens.get("cache")
        cache = cache if isinstance(cache, Mapping) else {}
        cost = payload.get("cost")
        cost = cost if isinstance(cost, Mapping) else {}
        amount = _amount(cost.get("amount", payload.get("cost_amount")))
        currency = cost.get("currency", payload.get("cost_currency"))
        currency = currency.upper() if isinstance(currency, str) and currency.strip() else None
        # A naked number has no currency.  Preserve it as unknown rather than
        # silently calling it USD, which would be a financial falsehood.
        if amount is not None and currency is None:
            amount = None
        return cls(
            id=new_id(RUNTIME_USAGE),
            session_id=session_id,
            run_id=run_id,
            runtime_binding_id=runtime_binding_id,
            provider_id=provider_id,
            model_id=model_id,
            reported_at=reported_at,
            input_tokens=_first_count((tokens.get("input"), tokens.get("input_tokens"))),
            output_tokens=_first_count((tokens.get("output"), tokens.get("output_tokens"))),
            reasoning_tokens=_first_count((tokens.get("reasoning"), tokens.get("reasoning_tokens"))),
            cache_read_tokens=_first_count((cache.get("read"), tokens.get("cache_read_tokens"))),
            cache_write_tokens=_first_count((cache.get("write"), tokens.get("cache_write_tokens"))),
            total_tokens=_first_count((tokens.get("total"), tokens.get("total_tokens"))),
            cost_amount=amount,
            cost_currency=currency,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "usage_event_id": self.id,
            "runtime_binding_id": self.runtime_binding_id,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "reported_at": self.reported_at.isoformat(),
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "reasoning": self.reasoning_tokens,
                "cache_read": self.cache_read_tokens,
                "cache_write": self.cache_write_tokens,
                "total": self.total_tokens,
            },
            "cost": {
                "amount": str(self.cost_amount) if self.cost_amount is not None else None,
                "currency": self.cost_currency,
            },
        }
