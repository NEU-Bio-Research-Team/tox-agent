"""EvalTraceV1: one canonical trace projection for every driver.

Trajectory graders used to read whatever a driver happened to put on
:class:`TaskOutcome` — a list of tool names, mostly. A grader written against a
scripted run's shape and a grader written against a live run's shape drift
apart, and neither can be replayed against a production run.

The projection here is built only from what the product API returns (the run
projection with its tool calls and runtime binding, the answer, the evidence,
the decision-support state when present), so the same function serves the
scripted driver, a live stack and a sanitized production replay. It holds no
chain-of-thought and no tool arguments — an argument hash is enough to see a
repeated call.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from evals.graders.model import TaskOutcome

SCHEMA_VERSION = "eval-trace-v1"

SEARCH_TOOLS = frozenset({"search_toxicology_evidence"})
READ_EVIDENCE_TOOLS = frozenset({"get_evidence_record"})
SUBMIT_TOOLS = frozenset(
    {"submit_grounded_answer", "submit_report_draft", "submit_saved_report_draft",
     "submit_report_synthesis"}
)


@dataclass(frozen=True)
class TraceToolEvent:
    index: int
    tool_name: str
    status: str
    error_code: str | None
    arguments_sha256: str | None
    duration_ms: int | None
    observation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalTraceV1:
    run_id: str | None
    intent: str | None
    lane: str | None
    run_status: str | None
    failure_code: str | None
    capability_profile: str | None
    runtime_agent: str | None
    tools: tuple[TraceToolEvent, ...]
    #: first_pass | accepted_after_correction | fallback | none
    answer_outcome: str
    candidate_generation: int | None
    is_fallback: bool | None
    evidence_ids: tuple[str, ...]
    stop_reason: str | None = None
    coverage: dict[str, Any] | None = None
    usage_status: str | None = None
    counters: dict[str, int] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def project(outcome: TaskOutcome) -> EvalTraceV1:
    run = outcome.run or {}
    runtime = run.get("runtime") or {}
    tools = tuple(
        TraceToolEvent(
            index=i,
            tool_name=call.get("tool_name") or "",
            status=call.get("status") or "",
            error_code=call.get("error_code"),
            arguments_sha256=call.get("arguments_sha256"),
            duration_ms=call.get("duration_ms"),
            observation_ids=tuple(call.get("observation_ids") or ()),
        )
        for i, call in enumerate(outcome.tool_calls)
    )
    answer = outcome.answer
    if answer is None:
        answer_outcome, generation, fallback = "none", None, None
    else:
        generation = answer.get("candidate_generation")
        fallback = bool(answer.get("is_fallback"))
        if fallback:
            answer_outcome = "fallback"
        elif (generation or 1) > 1:
            answer_outcome = "accepted_after_correction"
        else:
            answer_outcome = "first_pass"

    seen: dict[tuple[str, str | None], int] = {}
    duplicates = 0
    for event in tools:
        if event.arguments_sha256 is None:
            continue
        key = (event.tool_name, event.arguments_sha256)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            duplicates += 1

    state = outcome.decision_state or {}
    counters = {
        "tool_calls": len(tools),
        "searches": sum(1 for t in tools if t.tool_name in SEARCH_TOOLS),
        "evidence_reads": sum(1 for t in tools if t.tool_name in READ_EVIDENCE_TOOLS),
        "submits": sum(1 for t in tools if t.tool_name in SUBMIT_TOOLS),
        "failed_calls": sum(1 for t in tools if t.status not in ("completed", "")),
        # Visibility, budget and loop denials share one code on purpose
        # (tools/runner.py: a model must not be able to tell them apart).
        "denied_calls": sum(1 for t in tools if t.error_code == "tool_denied"),
        "duplicate_calls": duplicates,
    }
    return EvalTraceV1(
        run_id=run.get("run_id"),
        intent=run.get("intent"),
        lane=run.get("lane"),
        run_status=run.get("status"),
        failure_code=run.get("failure_code"),
        capability_profile=runtime.get("capability_profile"),
        runtime_agent=runtime.get("runtime_agent_name"),
        tools=tools,
        answer_outcome=answer_outcome,
        candidate_generation=generation,
        is_fallback=fallback,
        evidence_ids=tuple(sorted(i for i in outcome.session_evidence_ids if i)),
        stop_reason=state.get("stop_reason"),
        coverage=state.get("coverage"),
        usage_status=(run.get("usage") or {}).get("status"),
        counters=counters,
    )
