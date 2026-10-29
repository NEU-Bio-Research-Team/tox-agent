"""What the investigator reads before each step (#63, §5.4).

The case already had a read model, ``compile_dossier``, and it is the wrong
one for this job: it is the *product* of a run, the thing chat, report and UI
are views of. §5.4 asks for the other direction -- what goes *into* the turn --
and names six groups, with a rule about each of them that the dossier does not
follow:

1. the task and subject in force, and the facts the contract requires;
2. the observations that bear on it, **with their identifiers**;
3. the conclusions in force, and what just changed;
4. the evidence, the counter-evidence, and the metadata that is missing;
5. the unresolved gaps, what was already searched, and the budget left;
6. the instructions and skills for *this* task.

And the rule that makes it a view rather than a dump: raw full text, hundreds
of assay rows and the whole history stay out of the prompt. This returns
counts and refs; the tools read the content. A view that carried the rows
would spend the context window on what a single tool call can fetch, and §I3
is explicit that tool output belongs behind a reference.

The three kinds of memory §5.4 separates are kept apart here too, because
they have different lifetimes and different owners:

* **procedural** -- reviewed skills, versioned, the same for every case;
* **episodic** -- this case's history, which only the researcher's subject
  scopes;
* **execution** -- this run's step results, which resume reads and a later
  turn must not mistake for findings.

A summary may help a turn navigate; it may never stand in for raw evidence in
verification. So nothing in here is citable: every value a claim quotes is
read through a tool, from an artifact with an id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ...domain.scientific_case import (
    CLAIM_PROPOSED,
    CLAIM_RELEASED,
    ScientificCaseV1,
    Stance,
    needs_reassessment,
    value_versions,
)
from .goal_contract import GoalContractV1
from .researcher_values import citable_field

CASE_VIEW_SCHEMA_VERSION = "case-view-v1"

#: How many of each group reach the prompt. The rest are counted, not shown:
#: a turn that needs the 40th ledger entry reads it with a tool.
DEFAULT_LIMIT = 6


@dataclass(frozen=True, slots=True)
class CaseView:
    """The six groups, each already bounded."""

    subject: Mapping[str, Any] = field(default_factory=dict)
    contract: Mapping[str, Any] = field(default_factory=dict)
    observations: tuple[Mapping[str, Any], ...] = ()
    conclusions: Mapping[str, Any] = field(default_factory=dict)
    changed: tuple[str, ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()
    counter_evidence: tuple[Mapping[str, Any], ...] = ()
    missing_metadata: tuple[str, ...] = ()
    gaps: tuple[Mapping[str, Any], ...] = ()
    searched: tuple[str, ...] = ()
    budget_left: Mapping[str, Any] = field(default_factory=dict)
    #: Group 6, and the procedural half of the memory split.
    skills: tuple[Mapping[str, Any], ...] = ()
    #: What was left out, by group, so the turn knows a tool can fetch more.
    withheld: Mapping[str, int] = field(default_factory=dict)
    schema_version: str = CASE_VIEW_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subject": dict(self.subject),
            "contract": dict(self.contract),
            "observations": [dict(o) for o in self.observations],
            "conclusions": dict(self.conclusions),
            "changed": list(self.changed),
            "evidence": [dict(e) for e in self.evidence],
            "counter_evidence": [dict(e) for e in self.counter_evidence],
            "missing_metadata": list(self.missing_metadata),
            "gaps": [dict(g) for g in self.gaps],
            "searched": list(self.searched),
            "budget_left": dict(self.budget_left),
            "skills": [dict(s) for s in self.skills],
            "withheld": dict(self.withheld),
        }

    def render(self) -> str:
        """The view as prompt text, in §5.4's order."""
        lines: list[str] = []
        if self.contract:
            lines.append(GoalContractV1.from_dict(self.contract).render())
        if self.observations:
            lines.append("Observations in this case (read values with a tool, never from here):")
            lines.extend(_observation_line(o) for o in self.observations)
            self._withheld_line(lines, "observations")
        if self.conclusions.get("can_say") or self.conclusions.get("cannot_say"):
            said = "; ".join(self.conclusions.get("can_say") or [])
            not_said = "; ".join(self.conclusions.get("cannot_say") or [])
            status = self.conclusions.get("status", "")
            lines.append(
                f"Conclusion in force ({status}): " + (said or "nothing yet")
                + (f". Explicitly not established: {not_said}" if not_said else "")
            )
        if self.conclusions.get("derived"):
            lines.append("Derived claims in force (computed by the server from cited inputs):")
            lines.extend(f"- {item}" for item in self.conclusions["derived"])
        if self.changed:
            lines.append("Changed since the last assessment, and so not settled any more:")
            lines.extend(f"- {item}" for item in self.changed)
        if self.evidence:
            lines.append("Evidence on file:")
            lines.extend(f"- {e['ref']} {e.get('what', '')}".rstrip() for e in self.evidence)
            self._withheld_line(lines, "evidence")
        if self.counter_evidence:
            lines.append("Counter-evidence on file (do not drop it from the answer):")
            lines.extend(
                f"- {e['ref']} {e.get('what', '')}".rstrip() for e in self.counter_evidence
            )
        if self.missing_metadata:
            lines.append(
                "Metadata these records do not state -- unknown, not a match: "
                + "; ".join(self.missing_metadata)
            )
        if self.gaps:
            lines.append("Unresolved gaps. Choose the one that most changes the answer:")
            lines.extend(
                f"- [{g.get('gap', '?')}] {g.get('question', '')}".rstrip() for g in self.gaps
            )
        if self.searched:
            lines.append(
                "Already searched this subject (do not repeat these queries): "
                + "; ".join(self.searched)
            )
        if self.budget_left:
            spent = "; ".join(f"{k}: {v}" for k, v in sorted(self.budget_left.items()))
            lines.append(f"Budget left — {spent}.")
        return "\n".join(lines)

    def _withheld_line(self, lines: list[str], group: str) -> None:
        held = self.withheld.get(group, 0)
        if held:
            lines.append(f"  (+{held} more in the case; read them with a tool)")


def _observation_line(row: Mapping[str, Any]) -> str:
    line = f"- {row['ref']} {row.get('what', '')}".rstrip()
    if row.get("source_class") != "researcher_value":
        return line
    line += f" [{row['value_ref']}"
    if not row.get("current", True):
        line += ", superseded -- cite the version in force"
    elif row.get("observation_id"):
        line += f"; cite observation_id {row['observation_id']} field_path {row['field_path']}"
    return line + "]"


def _observation_rows(
    case: ScientificCaseV1, limit: int, researcher_observations: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Predictor, policy and workflow facts the case already holds, by ref."""
    rows = [
        {"ref": entry.source_ref or entry.id, "what": entry.claim[:160],
         "source_class": entry.source_class}
        for entry in case.evidence
        if entry.source_class in ("predictor_fact", "explanation_fact", "report_fact")
    ]
    # The version ref is what a claim computed from the value cites, and a
    # superseded version stays listed: the turn has to see that the number it
    # read last time is no longer the one in force (§11 turn 3).
    researcher_observations = researcher_observations or {}
    researcher = [
        {"ref": f"context:{version.item_id}", "value_ref": version.ref,
         "what": item.describe()[:160], "source_class": "researcher_value",
         "current": version.current,
         # What an answer cites to quote the value: the observation this run
         # projected it into (#61). Only a version in force has one.
         **({"observation_id": researcher_observations[version.ref],
             "field_path": citable_field(version)}
            if version.ref in researcher_observations else {})}
        for item, version in zip(case.context, value_versions(case))
    ]
    everything = researcher + rows
    return everything[:limit], max(0, len(everything) - limit)


def _evidence_rows(
    case: ScientificCaseV1, limit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    supporting, against = [], []
    for entry in case.evidence:
        if entry.source_class in ("predictor_fact", "explanation_fact", "report_fact"):
            continue
        row = {"ref": entry.id, "what": f"[{entry.stance}] {entry.claim[:140]}",
               "source_ref": entry.source_ref}
        (against if entry.stance == Stance.CONTRADICTS.value else supporting).append(row)
    held = max(0, len(supporting) - limit)
    return supporting[:limit], against[:limit], held


#: The axes that decide whether two records measured the same thing, among
#: the ones a ledger entry can carry. A record silent on one of these is not
#: comparable on it, and the turn has to see the silence: ``scope`` drops
#: empty values when it is written, so an unstated axis is an *absent* key,
#: never an empty one.
_DECISIVE_AXES: tuple[str, ...] = (
    "target_id", "target_variant", "assay", "assay_type", "readout", "species",
    "experimental_context",
)


def _missing_metadata(case: ScientificCaseV1, limit: int) -> tuple[str, ...]:
    """Axes a recorded relation did not declare. §5.2: unknown is unknown, and
    a turn that cannot see what is unstated will read the silence as agreement.

    Only reported for an entry that declared *something*: an entry with no
    scope at all is not a record missing three fields, it is a record nobody
    scoped, and saying "missing: all seven" of it would bury the entries where
    one decisive axis is genuinely absent.
    """
    notes: list[str] = []
    for entry in case.evidence:
        if not entry.scope:
            continue
        unstated = [axis for axis in _DECISIVE_AXES if not entry.scope.get(axis)]
        if unstated and len(unstated) < len(_DECISIVE_AXES):
            notes.append(f"{entry.id}: {', '.join(unstated)}")
    return tuple(notes[:limit])


def build(
    case: ScientificCaseV1 | None,
    *,
    contract: GoalContractV1 | None = None,
    open_gaps: Sequence[Mapping[str, Any]] = (),
    searched: Sequence[str] = (),
    budget_left: Mapping[str, Any] | None = None,
    skills: Sequence[Mapping[str, Any]] = (),
    limit: int = DEFAULT_LIMIT,
    researcher_observations: Mapping[str, str] | None = None,
) -> CaseView:
    """The view for one step. Pure: same case and state, same view.

    ``case`` may be ``None`` -- the first turn of a subject has no case yet,
    and a contract plus a budget is still a view worth sending.
    """
    if case is None:
        return CaseView(
            contract=contract.to_dict() if contract else {},
            gaps=tuple(dict(g) for g in open_gaps),
            searched=tuple(dict.fromkeys(searched)),
            budget_left=dict(budget_left or {}),
            skills=tuple(dict(s) for s in skills),
        )

    observations, held_observations = _observation_rows(case, limit, researcher_observations)
    evidence, counter, held_evidence = _evidence_rows(case, limit)
    stale = needs_reassessment(case.derived_claims)
    trusted = case.conclusion if case.conclusion.status == "released" else case.released_conclusion
    return CaseView(
        subject=dict(case.subject),
        contract=contract.to_dict() if contract else {},
        observations=tuple(observations),
        conclusions={
            "status": case.conclusion.status,
            "can_say": [line.text[:160] for line in (trusted.can_say if trusted else ())],
            "cannot_say": [text[:160] for text in (trusted.cannot_say if trusted else ())],
            "derived": [
                f"{claim.id} [{claim.status}] {claim.statement[:140]}"
                for claim in case.derived_claims
                if claim.status in (CLAIM_PROPOSED, CLAIM_RELEASED)
            ][:limit],
        },
        changed=tuple(
            f"{claim.id} {claim.statement[:120]} — {claim.note[:140]}" for claim in stale[:limit]
        ),
        evidence=tuple(evidence),
        counter_evidence=tuple(counter),
        missing_metadata=_missing_metadata(case, limit),
        gaps=tuple(dict(g) for g in open_gaps),
        searched=tuple(dict.fromkeys(searched)),
        budget_left=dict(budget_left or {}),
        skills=tuple(dict(s) for s in skills),
        withheld={"observations": held_observations, "evidence": held_evidence},
    )
