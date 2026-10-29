"""The goal contract a run is admitted under (#69, §6.1).

§6.1 turns the researcher's message into five declared things: the subject and
endpoint, the questions that must be answered, the facts needed to answer
them, the actions allowed, and the budget with its stop conditions. Before
this, four of the five existed and none of them existed *together*: the
subject was on the run binding, the budget in ``EffectiveRunBudgetV1``, the
allowed actions implied by whichever capability profile the registry handed
out, and the stop conditions in an enum the model never saw.

So nothing could answer the question a grader asks first -- what was this run
*asked* to do, and was it allowed to do it -- without reconstructing it from
four places and a prompt.

Two rules the module is built around:

* **Required claims come from the question, never from gold.** The contract is
  a projection of the researcher's message, the bound subject, the route, the
  registered tools and the admitted budget. A benchmark case's expected
  answer is not among its inputs, and :func:`build` has no parameter that
  could carry one. The test that matters asserts exactly that.
* **A contract is not a plan.** It says what must be answered, not how. The
  propositions a run opens are the model's (``decision_state.apply_plan``),
  and the actions listed here are the ones the deployment actually registered
  -- an allowlist, not an invitation to the whole tool registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ...domain.investigation_gap import Gap

GOAL_CONTRACT_SCHEMA_VERSION = "goal-contract-v1"

#: The stop conditions a decision-support run may end on, as the contract
#: states them to the model. The values are ``decision_state.StopReason``'s;
#: the wording is what a run has to be able to say about itself.
STOP_CONDITIONS: Mapping[str, str] = {
    "sufficient": "every question this contract names is resolved, in scope, with refs",
    "needs_user_input": "an input only the researcher has, named explicitly",
    "provider_failure": "a provider or corpus could not supply the data; say which",
    "insufficient_evidence": "the reachable sources do not settle it; say what is missing "
                             "and which claim that limits",
    "budget_exhausted": "a declared limit was reached with a question still open",
}


@dataclass(frozen=True, slots=True)
class RequiredQuestion:
    """One thing the run must answer, and what it takes to answer it."""

    id: str
    question: str
    #: Kinds of source an answer to this needs: ``prediction``, ``evidence``,
    #: ``report``, ``explanation``, ``researcher_input``.
    needs: tuple[str, ...] = ()
    #: Which gap stands between the run and this answer right now, when the
    #: server can tell. Empty means the run has to work that out.
    gap: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id, "question": self.question,
                                "needs": list(self.needs)}
        if self.gap:
            data["gap"] = self.gap
        return data


@dataclass(frozen=True, slots=True)
class GoalContractV1:
    """What this run was asked to do, and what it may do about it."""

    route: str
    subject: Mapping[str, Any] = field(default_factory=dict)
    endpoint: str = ""
    must_answer: tuple[RequiredQuestion, ...] = ()
    #: Tool names the run may call, as the registry resolved them for this
    #: turn's capability profile. Not a wish list: a name here exists.
    admissible_actions: tuple[str, ...] = ()
    #: The admitted ``EffectiveRunBudgetV1``, as recorded.
    budget: Mapping[str, Any] = field(default_factory=dict)
    stop_conditions: Mapping[str, str] = field(default_factory=lambda: dict(STOP_CONDITIONS))
    #: The researcher-value versions in force when this contract was written.
    #: The next run of the subject diffs against it to know what is new (§4.1
    #: update route) -- a fact the server holds, not a reading of the wording.
    researcher_values_seen: tuple[str, ...] = ()
    #: The compare route's candidates, one subject each (§4.1 compare/batch).
    candidates: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = GOAL_CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "route": self.route,
            "subject": dict(self.subject),
            "endpoint": self.endpoint,
            "must_answer": [q.to_dict() for q in self.must_answer],
            "admissible_actions": list(self.admissible_actions),
            "budget": dict(self.budget),
            "stop_conditions": dict(self.stop_conditions),
        }
        if self.researcher_values_seen:
            data["researcher_values_seen"] = list(self.researcher_values_seen)
        if self.candidates:
            data["candidates"] = [dict(c) for c in self.candidates]
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GoalContractV1":
        return cls(
            route=raw.get("route", ""),
            subject=dict(raw.get("subject") or {}),
            endpoint=raw.get("endpoint", ""),
            must_answer=tuple(
                RequiredQuestion(id=q["id"], question=q.get("question", ""),
                                 needs=tuple(q.get("needs") or ()), gap=q.get("gap", ""))
                for q in raw.get("must_answer") or ()
            ),
            admissible_actions=tuple(raw.get("admissible_actions") or ()),
            budget=dict(raw.get("budget") or {}),
            stop_conditions=dict(raw.get("stop_conditions") or STOP_CONDITIONS),
            researcher_values_seen=tuple(raw.get("researcher_values_seen") or ()),
            candidates=tuple(dict(c) for c in raw.get("candidates") or ()),
        )

    def render(self) -> str:
        """The contract as the turn reads it. Short on purpose: a contract the
        model skims is a contract it ignores."""
        lines = [f"Goal contract ({self.route} route)."]
        if self.subject:
            identity = self.subject.get("identity") or self.subject
            parts = [str(identity.get(key)) for key in ("canonical_smiles", "target", "assay_context")
                     if identity.get(key)]
            if parts:
                lines.append("Subject: " + "; ".join(parts) +
                             ". Evidence about another subject is not this run's evidence.")
        if self.endpoint:
            lines.append(f"Endpoint/context: {self.endpoint}")
        if self.must_answer:
            lines.append("You must answer, and nothing here is optional:")
            for question in self.must_answer:
                needs = f" (needs: {', '.join(question.needs)})" if question.needs else ""
                gap = (f" [open gap: {question.gap} -- {GAP_GUIDANCE[question.gap]}]"
                       if question.gap in GAP_GUIDANCE
                       else f" [open gap: {question.gap}]" if question.gap else "")
                lines.append(f"- {question.id} {question.question}{needs}{gap}")
        if self.candidates:
            lines.append("Candidates, each its own subject -- one row per candidate, never "
                         "a combined score: " + "; ".join(
                             str(c.get("subject_key")) for c in self.candidates))
        if self.admissible_actions:
            lines.append("Actions available to this turn: " +
                         ", ".join(self.admissible_actions) +
                         ". No other tool exists for you.")
        declared = _budget_lines(self.budget)
        if declared:
            lines.append("Budget: " + "; ".join(declared) +
                         ". A review and a repair are paid for out of this.")
        lines.append(
            "Stop when one of these is true, and say which: "
            + "; ".join(f"{name} — {meaning}" for name, meaning in self.stop_conditions.items())
            + "."
        )
        return "\n".join(lines)


def _budget_lines(budget: Mapping[str, Any]) -> list[str]:
    """The limits worth stating, in the order a run spends them."""
    named = (
        ("deadline_s", "seconds"), ("max_tool_calls", "tool calls"),
        ("max_searches", "searches"), ("max_evidence_reads", "record reads"),
        ("max_full_text_reads", "full-text reads"), ("max_repairs", "repair"),
    )
    return [f"{budget[key]} {label}" for key, label in named if budget.get(key) is not None]


#: What a calculation the server performs is made of. A requirement table,
#: not a router: it does not decide how a question is investigated, only what
#: an answer to it cannot do without -- so a margin question whose case holds
#: no measured potency is told that, instead of reading the predictor's label
#: and stopping (live, arm E on T5, 2026-10-29: four turns, no search). Only
#: the margin is here because it is the only calculation the server has.
_CALCULATION_MARKERS: tuple[str, ...] = ("margin",)


def requirements(
    question: str, *, has_measurement: bool, has_exposure: bool,
) -> tuple[tuple[str, ...], str]:
    """``(needs, gap)`` for the question a run was asked (§6.2 steps 1-2).

    The gap is the most important unmet requirement the run can act on: a
    missing measurement before a missing exposure, because the run can search
    for the first and only the researcher has the second -- and a run that
    stops at "give me your Cmax" without looking for the potency has to be
    asked twice.
    """
    lowered = question.lower()
    if not any(marker in lowered for marker in _CALCULATION_MARKERS):
        return (), ""
    needs = ("evidence", "researcher_input")
    if not has_measurement:
        return needs, Gap.NO_RELEVANT_EVIDENCE.value
    if not has_exposure:
        return needs, Gap.ONLY_RESEARCHER_HAS_IT.value
    return needs, ""


#: What an open gap asks of the turn, in one line the contract prints.
GAP_GUIDANCE: dict[str, str] = {
    Gap.NO_RELEVANT_EVIDENCE.value: "search for a measured value before stopping; a "
                                    "predicted label or probability is not a measurement",
    Gap.ONLY_RESEARCHER_HAS_IT.value: "name the input and stop with needs_user_input",
    Gap.MEASUREMENT_UNRESOLVED.value: "recompute from the inputs in force, or say why not",
}


def build(
    *,
    route: str,
    question: str,
    question_needs: Sequence[str] = (),
    question_gap: str = "",
    subject: Mapping[str, Any] | None = None,
    endpoint: str = "",
    admissible_actions: Iterable[str] = (),
    budget: Mapping[str, Any] | None = None,
    open_questions: Sequence[Mapping[str, Any]] = (),
    stale_claims: Sequence[str] = (),
    new_values: Sequence[Mapping[str, Any]] = (),
    researcher_values_seen: Sequence[str] = (),
    candidates: Sequence[Mapping[str, Any]] = (),
    clarification: str = "",
) -> GoalContractV1:
    """The contract for one run.

    ``open_questions`` are propositions carried over from an earlier turn of
    the same subject; ``stale_claims`` are derived claims a changed premise
    left waiting (#60). Both turn into things that must be answered, which is
    what makes the update route (§4.1) a contract rather than a fresh start.

    There is deliberately no ``gold`` or ``expected`` parameter. §6.1: required
    claims come from the question.
    """
    must: list[RequiredQuestion] = []
    if clarification:
        # §4.2: the undetermined route asks. Put first, because answering the
        # question below without it is guessing.
        must.append(RequiredQuestion(
            id="q0", question=clarification, needs=("researcher_input",),
            gap=Gap.ONLY_RESEARCHER_HAS_IT.value,
        ))
    if question.strip():
        must.append(RequiredQuestion(id="q1", question=question.strip()[:500],
                                     needs=tuple(question_needs), gap=question_gap))
    for index, carried in enumerate(open_questions, start=len(must) + 1):
        text = str(carried.get("question") or "").strip()
        if not text:
            continue
        must.append(RequiredQuestion(
            id=f"q{index}", question=text[:500],
            needs=tuple(carried.get("required_sources") or ()),
            gap=str(carried.get("gap") or ""),
        ))
    for index, claim_id in enumerate(stale_claims, start=len(must) + 1):
        must.append(RequiredQuestion(
            id=f"q{index}",
            question=f"Derived claim {claim_id} rests on a premise that changed: recompute it "
                     "or say it cannot be recomputed.",
            needs=("prediction", "researcher_input"),
            gap=Gap.MEASUREMENT_UNRESOLVED.value,
        ))
    for value in new_values:
        # §4.1 update route: what the researcher just supplied, by version,
        # is something the run has to account for -- including saying what
        # it leaves unchanged.
        unit = f" {value['unit']}" if value.get("unit") else ""
        must.append(RequiredQuestion(
            id=f"q{len(must) + 1}",
            question=f"The researcher supplied {value['key']} = {value['value']}{unit} "
                     f"({value['ref']}). Say what it changes in the assessment and what it "
                     "leaves as it was.",
            needs=("researcher_input",),
        ))
    if route == "compare_batch" and len(candidates) > 1:
        must.append(RequiredQuestion(
            id=f"q{len(must) + 1}",
            question="Compare the candidates by the one criterion the question declares, "
                     "with run_scientific_procedure compare_candidates. Each row comes from "
                     "that candidate's own artifacts; do not combine criteria into a score. "
                     "If the question declares no criterion, ask which one.",
            needs=("prediction",),
        ))
    return GoalContractV1(
        route=route,
        researcher_values_seen=tuple(dict.fromkeys(researcher_values_seen)),
        candidates=tuple(dict(c) for c in candidates),
        subject=dict(subject or {}),
        endpoint=endpoint,
        must_answer=tuple(must),
        admissible_actions=tuple(dict.fromkeys(admissible_actions)),
        budget={k: v for k, v in dict(budget or {}).items() if v is not None},
    )
