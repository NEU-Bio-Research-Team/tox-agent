"""DecisionSupportStateV1: the goal, plan, coverage and stop of one ADS run.

Decision support deliberately does not run a fixed workflow (ADR 0010): the
model chooses what to read and whether to search. That choice left nothing
behind but a tool-call list, so nobody — not a resumed turn, not a benchmark —
could tell whether a run stopped because it had enough or because it ran out.

This is the smallest state that makes those facts data, and it is owned by the
server:

* the model may *propose* propositions (``apply_plan``); the server validates
  them and issues their ids;
* every tool result advances counters and records which artifacts became
  available (``apply_tool_result``);
* the accepted answer's evidence relations resolve propositions
  (``apply_answer``) — a proposition is ``supported`` or ``conflicted`` only
  with artifact refs behind it;
* the run's end fixes ``stop_reason`` (``finalize``).

Every transition is a pure function returning a new state with ``revision``
incremented, so a transition is replayable and testable without a database.
It is not a workflow: nothing here tells the model what to do next.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "decision-support-state-v1"

#: Upper bound on a plan. More than this is not a plan, it is a transcript.
MAX_PROPOSITIONS = 8


class PropositionStatus(str, Enum):
    OPEN = "open"
    SUPPORTED = "supported"
    #: Resolved against: the evidence establishes the negation.
    CONTRADICTED = "contradicted"
    #: Resolved as disputed: grounded sources on both sides.
    CONFLICTED = "conflicted"
    INSUFFICIENT = "insufficient"


class StopReason(str, Enum):
    #: Every proposition was resolved (supported, contradicted or conflicted)
    #: with refs.
    SUFFICIENT = "sufficient"
    #: The run answered, but at least one proposition stayed unresolved while
    #: budget remained — the model chose to stop; the answer must say what is
    #: missing.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    #: A limit was reached (a denied call, or searches/reads at their ceiling)
    #: while a proposition was still unresolved.
    BUDGET_EXHAUSTED = "budget_exhausted"
    #: The run ended without an accepted model answer (runtime failure,
    #: deadline, or the deterministic fallback).
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class RequiredSource(str, Enum):
    PREDICTION = "prediction"
    EXPLANATION = "explanation"
    EVIDENCE = "evidence"
    REPORT = "report"


#: Which tool results make which source kind available.
_SOURCE_BY_TOOL: Mapping[str, RequiredSource] = {
    "get_analysis_slice": RequiredSource.PREDICTION,
    "get_analysis_bundle": RequiredSource.PREDICTION,
    "get_explanation_slice": RequiredSource.EXPLANATION,
    "get_attribution": RequiredSource.EXPLANATION,
    "get_evidence_record": RequiredSource.EVIDENCE,
    "get_report_summary": RequiredSource.REPORT,
}

_RESOLVED = frozenset(
    {PropositionStatus.SUPPORTED.value, PropositionStatus.CONTRADICTED.value,
     PropositionStatus.CONFLICTED.value}
)

_SEARCH_TOOLS = frozenset({"search_toxicology_evidence"})
_READ_TOOLS = frozenset({"get_evidence_record"})

#: Relation source classes -> the ref prefix and required-source kind.
_REF_BY_SOURCE_CLASS: Mapping[str, tuple[str, RequiredSource | None]] = {
    "predictor_fact": ("observation", RequiredSource.PREDICTION),
    "explanation_fact": ("observation", RequiredSource.EXPLANATION),
    "external_experimental": ("evidence", RequiredSource.EVIDENCE),
    "external_regulatory": ("evidence", RequiredSource.EVIDENCE),
    "report_fact": ("report", RequiredSource.REPORT),
    "agent_synthesis": ("synthesis", None),
}


class InvalidPlan(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Proposition:
    id: str
    question: str
    required_sources: tuple[str, ...]
    status: str = PropositionStatus.OPEN.value
    artifact_refs: tuple[str, ...] = ()
    #: Where the proposition came from: the model's plan, or an answer's
    #: relation that named a proposition no plan had.
    origin: str = "plan"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "question": self.question,
            "required_sources": list(self.required_sources), "status": self.status,
            "artifact_refs": list(self.artifact_refs), "origin": self.origin,
        }


@dataclass(frozen=True, slots=True)
class DecisionSupportStateV1:
    session_id: str
    run_id: str
    goal: str
    subject_refs: tuple[str, ...]
    propositions: tuple[Proposition, ...] = ()
    budget_snapshot: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, int] = field(default_factory=dict)
    #: Artifacts this run read (and so may cite).
    available_refs: tuple[str, ...] = ()
    #: Evidence a search surfaced and accepted for this subject, not yet read.
    found_refs: tuple[str, ...] = ()
    stop_reason: str | None = None
    answer_outcome: str | None = None
    #: ADR 0012: which scientific skills this run was offered and which it
    #: actually read, each pinned by version and content hash, plus the arm
    #: (off/static/dynamic). Empty for a run that met no catalog.
    skills: Mapping[str, Any] = field(default_factory=dict)
    revision: int = 0
    schema_version: str = SCHEMA_VERSION

    @property
    def coverage(self) -> dict[str, int]:
        resolved = sum(1 for p in self.propositions if p.status in _RESOLVED)
        return {"required": len(self.propositions), "resolved": resolved}

    @property
    def unresolved(self) -> tuple[Proposition, ...]:
        return tuple(
            p for p in self.propositions
            if p.status in (PropositionStatus.OPEN.value, PropositionStatus.INSUFFICIENT.value)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "goal": self.goal,
            "subject_refs": list(self.subject_refs),
            "propositions": [p.to_dict() for p in self.propositions],
            "coverage": self.coverage,
            "budget_snapshot": dict(self.budget_snapshot),
            "usage": dict(self.usage),
            "available_refs": list(self.available_refs),
            "found_refs": list(self.found_refs),
            "stop_reason": self.stop_reason,
            "answer_outcome": self.answer_outcome,
            "skills": dict(self.skills),
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionSupportStateV1":
        return cls(
            session_id=data["session_id"], run_id=data["run_id"], goal=data.get("goal", ""),
            subject_refs=tuple(data.get("subject_refs") or ()),
            propositions=tuple(
                Proposition(
                    id=p["id"], question=p["question"],
                    required_sources=tuple(p.get("required_sources") or ()),
                    status=p.get("status", "open"),
                    artifact_refs=tuple(p.get("artifact_refs") or ()),
                    origin=p.get("origin", "plan"),
                )
                for p in data.get("propositions") or ()
            ),
            budget_snapshot=dict(data.get("budget_snapshot") or {}),
            usage=dict(data.get("usage") or {}),
            available_refs=tuple(data.get("available_refs") or ()),
            found_refs=tuple(data.get("found_refs") or ()),
            stop_reason=data.get("stop_reason"),
            answer_outcome=data.get("answer_outcome"),
            skills=dict(data.get("skills") or {}),
            revision=int(data.get("revision", 0)),
        )


# --------------------------------------------------------------- transitions

def _next(state: DecisionSupportStateV1, **changes: Any) -> DecisionSupportStateV1:
    return replace(state, revision=state.revision + 1, **changes)


def _guard_open(state: DecisionSupportStateV1) -> None:
    if state.stop_reason is not None:
        raise InvalidPlan(f"the state is final ({state.stop_reason}); it no longer changes")


def initial(
    *, session_id: str, run_id: str, goal: str, subject_refs: Iterable[str],
    budget_snapshot: Mapping[str, Any],
) -> DecisionSupportStateV1:
    return DecisionSupportStateV1(
        session_id=session_id, run_id=run_id, goal=goal.strip()[:2000],
        subject_refs=tuple(dict.fromkeys(subject_refs)),
        budget_snapshot=dict(budget_snapshot),
        usage={"tool_calls": 0, "searches": 0, "evidence_reads": 0, "denied_calls": 0,
               "failed_calls": 0},
        revision=1,
    )


def apply_plan(
    state: DecisionSupportStateV1, proposed: Sequence[Mapping[str, Any]],
) -> DecisionSupportStateV1:
    """Replace the open part of the plan with the model's proposal.

    Already-resolved propositions are kept: a re-plan cannot un-resolve what
    the run has established. Ids are issued here, never taken from input.
    """
    _guard_open(state)
    kept = tuple(p for p in state.propositions if p.status != PropositionStatus.OPEN.value)
    if len(kept) + len(proposed) > MAX_PROPOSITIONS:
        raise InvalidPlan(f"a plan holds at most {MAX_PROPOSITIONS} propositions")
    known_sources = {s.value for s in RequiredSource}
    seen_questions = {p.question.casefold() for p in kept}
    next_index = max((int(p.id[1:]) for p in state.propositions if p.id[1:].isdigit()), default=0)
    fresh: list[Proposition] = []
    for index, item in enumerate(proposed):
        question = str(item.get("question", "")).strip()
        if not question:
            raise InvalidPlan(f"propositions[{index}].question is empty")
        if len(question) > 500:
            raise InvalidPlan(f"propositions[{index}].question exceeds 500 characters")
        if question.casefold() in seen_questions:
            raise InvalidPlan(f"propositions[{index}] repeats an existing question")
        seen_questions.add(question.casefold())
        sources = tuple(dict.fromkeys(item.get("required_sources") or ()))
        unknown = sorted(set(sources) - known_sources)
        if unknown or not sources:
            raise InvalidPlan(
                f"propositions[{index}].required_sources must be a non-empty subset of "
                f"{sorted(known_sources)}; got {list(sources)}"
            )
        next_index += 1
        fresh.append(Proposition(id=f"p{next_index}", question=question, required_sources=sources))
    return _next(state, propositions=kept + tuple(fresh))


def apply_tool_result(
    state: DecisionSupportStateV1, *, tool_name: str, status: str,
    observation_ids: Sequence[str] = (), error_code: str | None = None,
    found_evidence_ids: Sequence[str] = (),
) -> DecisionSupportStateV1:
    if state.stop_reason is not None:
        return state
    usage = dict(state.usage)
    usage["tool_calls"] = usage.get("tool_calls", 0) + 1
    if tool_name in _SEARCH_TOOLS:
        usage["searches"] = usage.get("searches", 0) + 1
    if tool_name in _READ_TOOLS:
        usage["evidence_reads"] = usage.get("evidence_reads", 0) + 1
    available = list(state.available_refs)
    if status == "completed":
        kind = _SOURCE_BY_TOOL.get(tool_name)
        prefix = "evidence" if kind is RequiredSource.EVIDENCE else (
            "report" if kind is RequiredSource.REPORT else "observation"
        )
        if kind is not None:
            for identifier in observation_ids:
                ref = f"{prefix}:{identifier}"
                if ref not in available:
                    available.append(ref)
        found = list(state.found_refs)
        for identifier in found_evidence_ids:
            ref = f"evidence:{identifier}"
            if ref not in found:
                found.append(ref)
        return _next(state, usage=usage, available_refs=tuple(available), found_refs=tuple(found))
    elif error_code == "tool_denied":
        usage["denied_calls"] = usage.get("denied_calls", 0) + 1
    else:
        usage["failed_calls"] = usage.get("failed_calls", 0) + 1
    return _next(state, usage=usage, available_refs=tuple(available))


def apply_answer(
    state: DecisionSupportStateV1, *, relations: Sequence[Mapping[str, Any]],
    is_fallback: bool, candidate_generation: int,
) -> DecisionSupportStateV1:
    """Resolve propositions from the accepted answer's evidence relations.

    ``relations`` items: ``proposition`` (text), ``source_class``,
    ``source_id``, ``relation``. A relation that names a proposition the plan
    did not have adds one (origin ``answer``): an answer is allowed to resolve
    a question nobody wrote down, but the state must record that it did.
    """
    if state.stop_reason is not None:
        return state
    outcome = "fallback" if is_fallback else (
        "accepted_after_correction" if candidate_generation > 1 else "first_pass"
    )
    if is_fallback:
        # A server-authored fallback resolves nothing the model investigated.
        return _next(state, answer_outcome=outcome)
    by_question = {p.question.casefold(): p for p in state.propositions}
    order = [p.id for p in state.propositions]
    labels: dict[str, set[str]] = {}
    refs: dict[str, list[str]] = {}
    next_index = max((int(p.id[1:]) for p in state.propositions if p.id[1:].isdigit()), default=0)
    for item in relations:
        question = str(item.get("proposition", "")).strip()
        if not question:
            continue
        proposition = by_question.get(question.casefold())
        if proposition is None:
            next_index += 1
            prefix, kind = _REF_BY_SOURCE_CLASS.get(item.get("source_class", ""), ("artifact", None))
            proposition = Proposition(
                id=f"p{next_index}", question=question[:500],
                required_sources=(kind.value,) if kind else (), origin="answer",
            )
            by_question[question.casefold()] = proposition
            order.append(proposition.id)
        labels.setdefault(proposition.id, set()).add(str(item.get("relation")))
        prefix, _ = _REF_BY_SOURCE_CLASS.get(item.get("source_class", ""), ("artifact", None))
        if item.get("source_id") and item.get("source_class") != "agent_synthesis":
            refs.setdefault(proposition.id, []).append(f"{prefix}:{item['source_id']}")
    resolved: list[Proposition] = []
    by_id = {p.id: p for p in by_question.values()}
    for proposition_id in order:
        proposition = by_id[proposition_id]
        seen = labels.get(proposition_id, set())
        grounded = tuple(dict.fromkeys((*proposition.artifact_refs, *refs.get(proposition_id, ()))))
        if "supports" in seen and "contradicts" in seen and grounded:
            status = PropositionStatus.CONFLICTED.value
        elif "supports" in seen and grounded:
            status = PropositionStatus.SUPPORTED.value
        elif "contradicts" in seen and grounded:
            status = PropositionStatus.CONTRADICTED.value
        elif seen:
            # contextual/insufficient/not_applicable only, or a support with
            # nothing but synthesis behind it: looked at, not established.
            status = PropositionStatus.INSUFFICIENT.value
        else:
            status = proposition.status
        resolved.append(replace(proposition, status=status, artifact_refs=grounded))
    return _next(state, propositions=tuple(resolved), answer_outcome=outcome)


def record_skills_offered(
    state: DecisionSupportStateV1, *, mode: str, offered: Sequence[Mapping[str, str]],
) -> DecisionSupportStateV1:
    """The catalog arm and the skills this run was shown (RETHINK §4.8).

    In the static arm every offered skill was composed into the prompt, so it
    counts as read; in the dynamic arm only a read records one.
    """
    if state.stop_reason is not None:
        return state
    offered = [dict(item) for item in offered]
    return _next(state, skills={
        "mode": mode, "offered": offered,
        "loaded": list(offered) if mode == "static" else [],
        "references_loaded": [],
    })


def record_skill_loaded(
    state: DecisionSupportStateV1, *, pin: Mapping[str, str], reference: str | None = None,
) -> DecisionSupportStateV1:
    """A dynamic read of a skill body or one of its references. Idempotent."""
    if state.stop_reason is not None:
        return state
    skills = {
        "mode": state.skills.get("mode", "dynamic"),
        "offered": list(state.skills.get("offered") or ()),
        "loaded": list(state.skills.get("loaded") or ()),
        "references_loaded": list(state.skills.get("references_loaded") or ()),
    }
    if reference is None:
        if dict(pin) in skills["loaded"]:
            return state
        skills["loaded"].append(dict(pin))
    else:
        item = {**dict(pin), "reference": reference}
        if item in skills["references_loaded"]:
            return state
        skills["references_loaded"].append(item)
    return _next(state, skills=skills)


def _budget_reached(state: DecisionSupportStateV1) -> bool:
    usage, budget = state.usage, state.budget_snapshot
    if usage.get("denied_calls", 0) > 0:
        return True
    for used, limit in (("searches", "max_searches"), ("evidence_reads", "max_evidence_reads")):
        ceiling = budget.get(limit)
        if ceiling is not None and usage.get(used, 0) >= ceiling:
            return True
    ceiling = budget.get("max_tool_calls")
    return ceiling is not None and usage.get("tool_calls", 0) >= ceiling


def finalize(
    state: DecisionSupportStateV1, *, run_status: str,
) -> DecisionSupportStateV1:
    """Fix ``stop_reason`` from the state and how the run ended. Idempotent."""
    if state.stop_reason is not None:
        return state
    if run_status == "cancelled":
        reason = StopReason.CANCELLED
    elif run_status != "completed" or state.answer_outcome in (None, "fallback"):
        reason = StopReason.BLOCKED
    elif not state.unresolved:
        # No plan at all and an accepted answer with no relations is not
        # "sufficient" — nothing was established. It is an answer that did not
        # need evidence, recorded as such rather than as coverage.
        reason = StopReason.SUFFICIENT if state.propositions else StopReason.INSUFFICIENT_EVIDENCE
    elif _budget_reached(state):
        reason = StopReason.BUDGET_EXHAUSTED
    else:
        reason = StopReason.INSUFFICIENT_EVIDENCE
    return _next(state, stop_reason=reason.value)


def checkpoint_summary(state: DecisionSupportStateV1, *, limit: int = 4) -> str:
    """A compact, ref-bearing summary for the next turn's checkpoint."""
    lines = [f"Previous decision-support goal: {state.goal[:300]}"]
    if state.stop_reason:
        lines.append(f"It stopped: {state.stop_reason} (coverage {state.coverage['resolved']}/"
                     f"{state.coverage['required']}).")
    for proposition in state.propositions[:limit]:
        refs = ", ".join(proposition.artifact_refs[:3]) or "no refs"
        lines.append(f"- [{proposition.status}] {proposition.question[:200]} ({refs})")
    unresolved = [p for p in state.unresolved][:limit]
    if unresolved:
        lines.append("Still open: " + "; ".join(p.question[:120] for p in unresolved))
    return "\n".join(lines)

