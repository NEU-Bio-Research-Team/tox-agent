"""How an answer was released (#47).

An answer that passed validation is a candidate. Release is a separate,
recorded decision, because "passed validation", "reviewed and supported" and
"released although the review could not be used" are different facts, and a
grader, an auditor and a reader each need to know which one they are looking
at. Nothing here is a semantic pass unless the outcome says the review ran
and found nothing blocking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from .ids import ANSWER, RUN, SESSION, require_id


class ReleaseOutcome(str, Enum):
    #: The reviewer judged every claim under review and none blocks.
    REVIEWED = "reviewed"
    #: The first review blocked; the repaired answer was reviewed and passed.
    REPAIRED_AND_REVIEWED = "repaired_and_reviewed"
    #: The review still blocked after the one repair, or the repair submitted
    #: nothing usable. Released with the blocking claims named, never as
    #: reviewed: the reader is told which claims the reviewer did not accept.
    RELEASED_WITH_UNRESOLVED_REVIEW = "released_with_unresolved_review"
    #: The review could not act as a gate at all -- no reviewable claim, no
    #: verdicts, a reviewer that failed or ran out of time. The reason is kept.
    UNGATED = "ungated"
    #: The answer is the deterministic fallback; there is nothing to review.
    FALLBACK = "fallback"


#: Outcomes in which every claim under review was judged and none blocks.
SEMANTICALLY_PASSED = frozenset({ReleaseOutcome.REVIEWED, ReleaseOutcome.REPAIRED_AND_REVIEWED})


@dataclass(frozen=True, slots=True)
class AnswerRelease:
    answer_id: str
    run_id: str
    session_id: str
    outcome: ReleaseOutcome
    released_at: datetime
    #: Each review round as recorded: verdict counts, coverage, blocking ids.
    reviews: tuple[Mapping[str, Any], ...] = ()
    #: What the repair changed (release_gate.RepairEffect.to_dict), if one ran.
    repair: Mapping[str, Any] | None = None
    #: Claims the last review blocked that are still in the released answer.
    unresolved_claim_ids: tuple[str, ...] = ()
    #: Why the review could not gate, for UNGATED.
    ungated_reason: str | None = None
    #: The candidate the first review judged, when a repair replaced it.
    replaced_answer_id: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_id(self.answer_id, ANSWER, field="answer_release.answer_id")
        require_id(self.run_id, RUN, field="answer_release.run_id")
        require_id(self.session_id, SESSION, field="answer_release.session_id")
        if self.outcome is ReleaseOutcome.UNGATED and not self.ungated_reason:
            raise ValueError("an ungated release must say why the review could not gate it")
        if self.outcome is ReleaseOutcome.RELEASED_WITH_UNRESOLVED_REVIEW \
                and not self.unresolved_claim_ids and not self.ungated_reason:
            raise ValueError(
                "an answer released with an unresolved review must name the blocking claims "
                "or say why the repair produced nothing to review"
            )

    @property
    def semantically_passed(self) -> bool:
        return self.outcome in SEMANTICALLY_PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "outcome": self.outcome.value,
            "semantically_passed": self.semantically_passed,
            "reviews": [dict(r) for r in self.reviews],
            "repair": dict(self.repair) if self.repair is not None else None,
            "unresolved_claim_ids": list(self.unresolved_claim_ids),
            "ungated_reason": self.ungated_reason,
            "replaced_answer_id": self.replaced_answer_id,
            "released_at": self.released_at.isoformat(),
        }
