"""What each phase of a run spent, from what was recorded (#46/#53).

A decision-support run is several runtime turns under one parent run -- the
investigator (or arm B's synthesis), the reviewer, the one repair, the
reviewer again -- and a hard correction inside a turn. A total over the run
cannot say which of them cost what, and a cost table that reads only the
latest successful attempt hides what failed attempts spent.

Every dispatch records its phase on its runtime binding (``runtime_manifest
.phase``), and every tool call its phase on its row. This module groups the
recorded usage events by the phase of their binding, with the semantics the
normalizer recorded:

* cumulative reports: the latest revision per binding and provider message;
* deltas: added once;
* ``unknown`` semantics: kept out of every total and counted, so a phase with
  only such rows has unknown usage, not zero;
* a missing cost is unknown, never zero; costs in different currencies are
  reported side by side, never added.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any, Iterable, Mapping

#: The phases a dispatch or a tool call can belong to.
PHASES: tuple[str, ...] = (
    "investigator", "workflow_synthesis", "hard_correction", "review_initial",
    "semantic_repair", "review_final",
)

_TOKENS = ("input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens",
           "cache_write_tokens", "total_tokens")


def phase_of_binding(binding: Any) -> str:
    manifest = getattr(binding, "runtime_manifest", None) or {}
    return str(manifest.get("phase") or "unattributed")


def _contributing(events: Iterable[Any]) -> tuple[list[Any], int]:
    latest: dict[tuple[str, str], Any] = {}
    deltas: list[Any] = []
    unknown = 0
    for event in events:
        semantics = getattr(event, "semantics", "unknown")
        if semantics == "cumulative":
            key = (event.runtime_binding_id, event.provider_message_id or event.id)
            current = latest.get(key)
            if current is None or (event.revision or 0) >= (current.revision or 0):
                latest[key] = event
        elif semantics == "delta":
            deltas.append(event)
        else:
            unknown += 1
    return list(latest.values()) + deltas, unknown


def summarize_phase(events: list[Any]) -> dict[str, Any]:
    contributing, unknown = _contributing(events)
    tokens: dict[str, int | None] = {}
    for name in _TOKENS:
        values = [getattr(e, name) for e in contributing if getattr(e, name) is not None]
        tokens[name.removesuffix("_tokens")] = sum(values) if values else None
    costs: dict[str, Decimal] = defaultdict(Decimal)
    cost_unknown = 0
    for event in contributing:
        if event.cost_amount is None or not event.cost_currency:
            cost_unknown += 1
        else:
            costs[event.cost_currency.upper()] += event.cost_amount
    if not events:
        completeness = "not_measured"
    elif not contributing:
        completeness = "unknown"
    elif unknown or tokens["input"] is None or tokens["output"] is None:
        completeness = "partial"
    else:
        completeness = "complete"
    return {
        "tokens": tokens,
        "cost": {currency: str(amount) for currency, amount in sorted(costs.items())},
        "cost_completeness": (
            "not_measured" if not contributing else
            "complete" if not cost_unknown else "partial" if costs else "unknown"
        ),
        "completeness": completeness,
        "usage_events": len(events),
        "unknown_semantics_events": unknown,
        "models": sorted({e.model_id for e in contributing if e.model_id}),
    }


def phase_usage(*, bindings: Iterable[Any], usage_events: Iterable[Any],
                tool_calls: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Usage per phase, plus the tool calls each phase made (all attempts)."""
    binding_phase = {b.id: phase_of_binding(b) for b in bindings}
    by_phase: dict[str, list[Any]] = defaultdict(list)
    for event in usage_events:
        by_phase[binding_phase.get(event.runtime_binding_id, "unattributed")].append(event)
    calls: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for call in tool_calls:
        phase = str(call.get("phase") or "unattributed")
        calls[phase][str(call.get("status") or "unknown")] += 1
    phases = sorted(set(by_phase) | set(calls) | set(binding_phase.values()),
                    key=lambda p: (PHASES.index(p) if p in PHASES else len(PHASES), p))
    return {
        "schema_version": "phase-usage-v1",
        "phases": {
            phase: {
                **summarize_phase(by_phase.get(phase, [])),
                "dispatches": sum(1 for p in binding_phase.values() if p == phase),
                "tool_calls": dict(calls.get(phase, {})),
            }
            for phase in phases
        },
        "run_total": summarize_phase([e for events in by_phase.values() for e in events]),
    }
