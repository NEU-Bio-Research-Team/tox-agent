"""Derived claims and what they depend on (#60, §5.3).

A conclusion in this case used to be prose with a list of evidence ids and a
``what_would_change`` paragraph. That is enough to show a researcher why the
case says what it says, and not enough to answer the question §5.3 is about:
the researcher corrects the free Cmax, and *which* claims have to be computed
again? Before this, the only honest answers were "none" and "all of them".

So a derived claim records what the proposal's example records -- inputs, the
rule, the assumptions, a status -- and this module answers three questions
over that graph:

* **What does a change reach?** :func:`affected` walks inputs transitively: a
  margin depends on an exposure, and a recommendation depends on the margin.
* **Does a change invalidate the claim, or only one way of reaching it?** A
  claim may carry more than one justification, and losing one premise of one
  justification is not losing the claim. §5.3 asks for exactly this
  distinction, and for the weaker behaviour (mark it and move on) rather than
  deletion. A claim whose *every* justification is hit needs reassessment; a
  claim with one justification still standing keeps its status and records
  that the other one went.
* **Is the graph sound?** :func:`cycles` and :func:`unresolved_references`
  find the two ways it can stop meaning anything -- a claim that justifies
  itself, and an input nothing in the case provides.

What this is not: a truth maintenance system. §5.3 calls it "truth
maintenance có phạm vi" and the scope is the numeric and identity rules a
server can check. Nothing here decides whether a scientific interpretation is
right; it decides which interpretations have to be looked at again.

Statuses are deliberately not a boolean:

* ``proposed`` -- written by a run whose answer has not been published.
* ``released`` -- the case's trusted claim, matching ``CONCLUSION_RELEASED``.
* ``needs_reassessment`` -- every justification lost a premise. The claim and
  its number stay in the case, because deleting them would lose what the
  earlier answer was computed from, and a later answer still has to be able
  to say "this changed".
* ``retracted`` -- withdrawn by a run or the researcher, kept for the record.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping

#: Claim statuses. ``PROPOSED``/``RELEASED`` match the conclusion vocabulary
#: in ``model.py`` so one case does not carry two words for one state.
CLAIM_PROPOSED = "proposed"
CLAIM_RELEASED = "released"
CLAIM_NEEDS_REASSESSMENT = "needs_reassessment"
CLAIM_RETRACTED = "retracted"
CLAIM_STATUSES = (CLAIM_PROPOSED, CLAIM_RELEASED, CLAIM_NEEDS_REASSESSMENT, CLAIM_RETRACTED)

#: Scheme for an input that is another derived claim in the same case.
CLAIM_REF_SCHEME = "claim"


def claim_ref(claim_id: str) -> str:
    return f"{CLAIM_REF_SCHEME}:{claim_id}"


def is_claim_ref(ref: str) -> bool:
    return ref.startswith(f"{CLAIM_REF_SCHEME}:")


def referenced_claim_id(ref: str) -> str | None:
    return ref.split(":", 1)[1] if is_claim_ref(ref) else None


class Impact(str, Enum):
    """What a change did to one claim."""

    #: No justification of this claim used anything that changed.
    UNAFFECTED = "unaffected"
    #: At least one justification lost a premise, and at least one still
    #: stands. The claim is not in doubt; one route to it is.
    JUSTIFICATION_LOST = "justification_lost"
    #: Every justification lost a premise. The claim has to be computed or
    #: assessed again before it can be relied on.
    NEEDS_REASSESSMENT = "needs_reassessment"


@dataclass(frozen=True, slots=True)
class Justification:
    """One way of reaching a claim: a rule over inputs, under assumptions.

    ``inputs`` are references, never copied values: ``case_value:free_cmax@v1``
    for a researcher value, an observation or evidence id for an artifact,
    ``claim:dc1`` for another derived claim. A justification that copied the
    number would still look valid after the number changed, which is the bug
    this whole module exists to prevent.
    """

    id: str
    rule: str
    inputs: tuple[str, ...] = ()
    #: Stated conditions the rule needs and this case does not check, e.g.
    #: ``exposure_context_compatible``. Listed because an assumption nobody
    #: wrote down cannot be reviewed.
    assumptions: tuple[str, ...] = ()
    #: Set when a change took one of this justification's inputs away.
    invalidated_by: tuple[str, ...] = ()

    @property
    def standing(self) -> bool:
        return not self.invalidated_by

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"id": self.id, "rule": self.rule, "inputs": list(self.inputs)}
        if self.assumptions:
            data["assumptions"] = list(self.assumptions)
        if self.invalidated_by:
            data["invalidated_by"] = list(self.invalidated_by)
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Justification":
        return cls(
            id=raw["id"], rule=raw.get("rule", ""),
            inputs=tuple(raw.get("inputs") or ()),
            assumptions=tuple(raw.get("assumptions") or ()),
            invalidated_by=tuple(raw.get("invalidated_by") or ()),
        )


@dataclass(frozen=True, slots=True)
class DerivedClaim:
    """A claim the case derived, with every way it was reached."""

    id: str
    statement: str
    justifications: tuple[Justification, ...] = ()
    status: str = CLAIM_PROPOSED
    #: Why the status is what it is, in one line, for a reader who is not
    #: going to reconstruct the graph.
    note: str = ""
    actor: str = ""
    run_id: str | None = None

    @property
    def inputs(self) -> tuple[str, ...]:
        """Every input of every justification, in order, without duplicates."""
        seen: dict[str, None] = {}
        for justification in self.justifications:
            for ref in justification.inputs:
                seen.setdefault(ref, None)
        return tuple(seen)

    @property
    def standing_justifications(self) -> tuple[Justification, ...]:
        return tuple(j for j in self.justifications if j.standing)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id, "statement": self.statement,
            "justifications": [j.to_dict() for j in self.justifications],
            "status": self.status,
        }
        if self.note:
            data["note"] = self.note
        if self.actor:
            data["actor"] = self.actor
        if self.run_id:
            data["run_id"] = self.run_id
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DerivedClaim":
        return cls(
            id=raw["id"], statement=raw.get("statement", ""),
            justifications=tuple(Justification.from_dict(j)
                                 for j in raw.get("justifications") or ()),
            status=raw.get("status", CLAIM_PROPOSED), note=raw.get("note", ""),
            actor=raw.get("actor", ""), run_id=raw.get("run_id"),
        )


# ── soundness ─────────────────────────────────────────────────────────────


def unresolved_references(
    claims: Iterable[DerivedClaim], known: Iterable[str]
) -> tuple[tuple[str, str], ...]:
    """``(claim_id, ref)`` for every input nothing in the case provides.

    ``known`` is what the case can resolve: value refs, observation ids,
    evidence ids. Claim refs are resolved against ``claims`` itself. Returned
    as data because §5.3 asks for these to be *detected* -- an exception would
    make one stale reference hide the rest.
    """
    claims = tuple(claims)
    resolvable = set(known) | {claim_ref(c.id) for c in claims}
    return tuple(
        (claim.id, ref)
        for claim in claims
        for ref in claim.inputs
        if ref not in resolvable
    )


def cycles(claims: Iterable[DerivedClaim]) -> tuple[tuple[str, ...], ...]:
    """Every dependency cycle among the claims, each as the path that closes it.

    A claim that justifies itself has no premises outside itself, so no change
    ever reaches it and no reassessment ever finishes.
    """
    by_id = {claim.id: claim for claim in claims}
    found: list[tuple[str, ...]] = []
    seen_cycles: set[frozenset[str]] = set()

    def walk(claim_id: str, path: tuple[str, ...]) -> None:
        if claim_id in path:
            cycle = path[path.index(claim_id):] + (claim_id,)
            key = frozenset(cycle)
            if key not in seen_cycles:
                seen_cycles.add(key)
                found.append(cycle)
            return
        claim = by_id.get(claim_id)
        if claim is None:
            return
        for ref in claim.inputs:
            target = referenced_claim_id(ref)
            if target is not None:
                walk(target, path + (claim_id,))

    for claim_id in by_id:
        walk(claim_id, ())
    return tuple(found)


# ── what a change reaches ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ClaimImpact:
    claim_id: str
    impact: Impact
    #: The refs that reached this claim, directly or through another claim.
    through: tuple[str, ...] = ()
    #: Justifications that lost a premise, and those still standing.
    lost: tuple[str, ...] = ()
    standing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"claim_id": self.claim_id, "impact": self.impact.value,
                "through": list(self.through), "lost": list(self.lost),
                "standing": list(self.standing)}


def affected(
    claims: Iterable[DerivedClaim], changed: Iterable[str]
) -> tuple[ClaimImpact, ...]:
    """What ``changed`` reaches, claim by claim, transitively.

    Only claims that need reassessment propagate. A claim that lost one of two
    justifications is not itself in doubt, so it does not put everything
    downstream of it in doubt either -- propagating from it would make a
    single corrected exposure invalidate the whole case, which is the "all of
    them" answer this module replaced.
    """
    claims = tuple(claims)
    changed = tuple(dict.fromkeys(changed))
    impacts: dict[str, ClaimImpact] = {}
    invalid: set[str] = set(changed)

    # Repeat until nothing new needs reassessment: a claim's own invalidation
    # is what reaches the claims below it.
    for _ in range(len(claims) + 1):
        progressed = False
        for claim in claims:
            if impacts.get(claim.id) and impacts[claim.id].impact is Impact.NEEDS_REASSESSMENT:
                continue
            lost, standing, through = [], [], []
            for justification in claim.justifications:
                hit = tuple(ref for ref in justification.inputs if ref in invalid)
                if hit:
                    lost.append(justification.id)
                    through.extend(hit)
                else:
                    standing.append(justification.id)
            if not lost:
                impacts.setdefault(claim.id, ClaimImpact(claim.id, Impact.UNAFFECTED))
                continue
            impact = Impact.NEEDS_REASSESSMENT if not standing else Impact.JUSTIFICATION_LOST
            row = ClaimImpact(claim.id, impact, through=tuple(dict.fromkeys(through)),
                              lost=tuple(lost), standing=tuple(standing))
            if impacts.get(claim.id) != row:
                impacts[claim.id] = row
                progressed = True
            if impact is Impact.NEEDS_REASSESSMENT and claim_ref(claim.id) not in invalid:
                invalid.add(claim_ref(claim.id))
                progressed = True
        if not progressed:
            break

    return tuple(impacts[claim.id] for claim in claims if claim.id in impacts)


def apply_change(
    claims: Iterable[DerivedClaim], changed: Iterable[str], *, note: str = ""
) -> tuple[DerivedClaim, ...]:
    """The claims again, with statuses and justifications updated.

    Nothing is deleted. A claim that needs reassessment keeps its statement
    and its justifications, because the earlier answer was computed from them
    and a later answer has to be able to say what changed (§5.3). A retracted
    claim is left alone: it is already out of use, and re-opening it on a
    change would resurrect a claim somebody withdrew.
    """
    claims = tuple(claims)
    by_id = {impact.claim_id: impact for impact in affected(claims, changed)}
    updated: list[DerivedClaim] = []
    for claim in claims:
        impact = by_id.get(claim.id)
        if impact is None or impact.impact is Impact.UNAFFECTED:
            updated.append(claim)
            continue
        if claim.status == CLAIM_RETRACTED:
            updated.append(claim)
            continue
        justifications = tuple(
            replace(j, invalidated_by=tuple(dict.fromkeys(j.invalidated_by + impact.through)))
            if j.id in impact.lost else j
            for j in claim.justifications
        )
        if impact.impact is Impact.NEEDS_REASSESSMENT:
            reason = note or f"every justification lost a premise: {', '.join(impact.through)}"
            updated.append(replace(claim, justifications=justifications,
                                   status=CLAIM_NEEDS_REASSESSMENT, note=reason))
        else:
            reason = note or (
                f"justification {', '.join(impact.lost)} lost a premise; "
                f"{', '.join(impact.standing)} still stands"
            )
            updated.append(replace(claim, justifications=justifications, note=reason))
    return tuple(updated)


def needs_reassessment(claims: Iterable[DerivedClaim]) -> tuple[DerivedClaim, ...]:
    return tuple(c for c in claims if c.status == CLAIM_NEEDS_REASSESSMENT)
