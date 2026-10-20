"""Whether a reviewed answer may be released, and what a repair cost it.

Today the reviewer runs after the answer is accepted and only records
verdicts: "the answer stands and the run proceeds". The design asks for the
semantic check to happen *before* release, with one bounded repair -- an
unsupported claim has to be corrected, narrowed or removed.

This module is that decision, as a pure function, so the rule is testable on
its own and the harness wiring that calls it is a separate, smaller change.

Two things are decided here, and the second is why the first is safe:

**Release.** A ``not_supported`` verdict blocks: the claim says something its
sources do not. A ``partially_supported`` verdict blocks too, because the fix
is to narrow the claim to the scope its sources carry, and a claim that
overstates its scope is the specific error this project calls out.
``cannot_judge`` does not block -- the reviewer has no read tool and was given
a bundle, so it usually means the bundle was not enough, which is a fact about
the review and not about the claim.

**What the repair cost.** A system can pass any support check by deleting
every claim. Factual precision goes up, the answer stops being useful, and the
final pass rate alone cannot tell the difference. So a repair is measured:
what was removed, what was narrowed, what was added, and whether a required
limitation was lost on the way. A repair that only deletes is reported as
such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ...domain.errors import Violation

#: Verdicts that stop an answer being released as written.
BLOCKING_VERDICTS: frozenset[str] = frozenset({"not_supported", "partially_supported"})

#: What the model is asked to do about each blocking verdict. An unsupported
#: claim has three honest exits and this names them, so "say it more
#: confidently" is not among the options.
_REMEDIES: dict[str, str] = {
    "not_supported": (
        "its sources do not state this: correct it to what they do state, or remove it"
    ),
    "partially_supported": (
        "its sources state this for a narrower or different scope: narrow the claim "
        "to that scope, or remove it"
    ),
}


@dataclass(frozen=True)
class ReleaseDecision:
    release: bool
    violations: tuple[Violation, ...] = ()
    blocking_claim_ids: tuple[str, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)
    #: Set when the review could not be used as a gate at all, e.g. no
    #: verdicts arrived. The answer is released and the reason is recorded --
    #: a missing reviewer must not silently become a passing review.
    ungated_reason: str | None = None


def release_decision(
    reviews: Sequence[Mapping[str, Any]],
    *,
    reviewed_claim_ids: Sequence[str],
) -> ReleaseDecision:
    """Whether this answer may be released as written.

    ``reviews`` are the reviewer's verdicts; ``reviewed_claim_ids`` is what the
    server put under review. A verdict for a claim that was not under review
    is ignored rather than trusted, and a claim under review with no verdict
    leaves the review incomplete -- which is recorded, not treated as support.
    """
    expected = list(reviewed_claim_ids)
    if not expected:
        return ReleaseDecision(
            release=True, ungated_reason="the answer makes no reviewable claim"
        )
    by_claim = {
        str(item.get("claim_id")): item
        for item in reviews
        if str(item.get("claim_id")) in set(expected)
    }
    if not by_claim:
        return ReleaseDecision(
            release=True, ungated_reason="the reviewer returned no usable verdict"
        )

    counts: dict[str, int] = {}
    violations: list[Violation] = []
    blocking: list[str] = []
    for claim_id in expected:
        item = by_claim.get(claim_id)
        if item is None:
            counts["no_verdict"] = counts.get("no_verdict", 0) + 1
            continue
        verdict = str(item.get("verdict") or "")
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict not in BLOCKING_VERDICTS:
            continue
        blocking.append(claim_id)
        reason = str(item.get("reason") or "").strip()
        violations.append(
            Violation(
                code=f"claim_{verdict}",
                message=(
                    f"the reviewer judged this claim {verdict.replace('_', ' ')}: "
                    f"{_REMEDIES[verdict]}"
                    + (f" Reviewer's reason: {reason}" if reason else "")
                ),
                path=f"claims[{claim_id}]",
            )
        )
    return ReleaseDecision(
        release=not violations,
        violations=tuple(violations),
        blocking_claim_ids=tuple(blocking),
        counts=counts,
    )


def _claim_texts(answer: Any) -> dict[str, str]:
    return {c.claim_id: c.text for c in getattr(answer, "claims", ())}


def _limitation_codes(answer: Any) -> set[str]:
    codes: set[str] = set()
    for limitation in getattr(answer, "limitations", ()):
        code = getattr(limitation, "code", None) or getattr(limitation, "kind", None)
        codes.add(str(getattr(code, "value", code)))
    return {c for c in codes if c and c != "None"}


@dataclass(frozen=True)
class RepairEffect:
    """What one repair turn did to the answer, beyond whether it passed.

    Reported alongside the pass rate because a repair that only removes claims
    raises claim precision and lowers how much of the required information
    survived, and the pass rate alone shows the first and hides the second.
    """

    claims_before: int
    claims_after: int
    removed_claim_ids: tuple[str, ...]
    added_claim_ids: tuple[str, ...]
    reworded_claim_ids: tuple[str, ...]
    limitations_before: int
    limitations_after: int
    lost_limitation_codes: tuple[str, ...]
    relations_before: int
    relations_after: int

    @property
    def removal_only(self) -> bool:
        """Nothing was added and nothing reworded: the repair only deleted."""
        return bool(self.removed_claim_ids) and not (
            self.added_claim_ids or self.reworded_claim_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims_before": self.claims_before,
            "claims_after": self.claims_after,
            "removed_claim_ids": list(self.removed_claim_ids),
            "added_claim_ids": list(self.added_claim_ids),
            "reworded_claim_ids": list(self.reworded_claim_ids),
            "limitations_before": self.limitations_before,
            "limitations_after": self.limitations_after,
            "lost_limitation_codes": list(self.lost_limitation_codes),
            "relations_before": self.relations_before,
            "relations_after": self.relations_after,
            "removal_only": self.removal_only,
        }


def repair_effect(
    before: Any,
    after: Any,
    *,
    relations_before: int = 0,
    relations_after: int = 0,
) -> RepairEffect:
    """Compare the answer before a repair with the answer after it."""
    old, new = _claim_texts(before), _claim_texts(after)
    old_limitations, new_limitations = _limitation_codes(before), _limitation_codes(after)
    return RepairEffect(
        claims_before=len(old),
        claims_after=len(new),
        removed_claim_ids=tuple(sorted(set(old) - set(new))),
        added_claim_ids=tuple(sorted(set(new) - set(old))),
        reworded_claim_ids=tuple(
            sorted(cid for cid in set(old) & set(new) if old[cid] != new[cid])
        ),
        limitations_before=len(old_limitations),
        limitations_after=len(new_limitations),
        lost_limitation_codes=tuple(sorted(old_limitations - new_limitations)),
        relations_before=relations_before,
        relations_after=relations_after,
    )
