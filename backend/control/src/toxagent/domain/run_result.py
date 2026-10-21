"""What a decision-support run ended with, decided by one pure policy (#47).

Under the release gate an answer was released whatever the review found: an
answer the reviewer still blocked after its one repair went out labelled
``released_with_unresolved_review``, and an answer no reviewer could judge
went out ``ungated``. Both reached the assistant message, the answer route and
the case ledger. A label is not a gate.

``strict-reviewed-v1`` makes it one. Three things are kept apart, because they
are different facts and a reader needs each:

* **publication disposition** -- what the user is shown: ``publish`` (the
  model's answer, every reviewed claim judged supported), ``partial_facts``
  (no model answer; an incomplete result with the facts the server resolved
  itself and the gaps), or ``withhold`` (nothing could be shown);
* **review status** -- what the reviewer did, including that it could not;
* **execution and knowledge outcome** -- why the run stopped, on two axes, so
  a provider timeout is never recorded as "searched and found nothing".

Task success is none of these. It is graded against independent gold, never
inferred from a review.

``hard-only-v1`` is the declared ablation: no semantic review at all, the
hard-validated candidate is published, and the record says so. It is a mode,
not a reviewer made to time out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from .ids import ANSWER, RUN, SESSION, require_id
from .provenance import content_sha256

SCHEMA_VERSION = "run-result-v1"


class ReleasePolicy(str, Enum):
    #: Before #47's strict mode: the gate's flags decide, and an unresolved
    #: review is still released with a label. Records no run result.
    LEGACY = "legacy"
    STRICT_REVIEWED_V1 = "strict-reviewed-v1"
    #: The no-semantic-review ablation (proposal §4.8), declared up front.
    HARD_ONLY_V1 = "hard-only-v1"


class Disposition(str, Enum):
    PUBLISH = "publish"
    PARTIAL_FACTS = "partial_facts"
    WITHHOLD = "withhold"


class ReviewStatus(str, Enum):
    PASSED = "passed"
    REPAIRED_AND_PASSED = "repaired_and_passed"
    #: A claim was still judged not supported after the one repair, or the
    #: repair submitted nothing new.
    BLOCKED = "blocked"
    #: Some claims were judged cannot_judge or got no verdict; a partial
    #: review is not a pass.
    INCOMPLETE = "incomplete"
    #: The reviewer did not run, failed, timed out, or had no budget.
    NOT_COMPLETED = "not_completed"
    #: Nothing to review: no reviewable claim, or a server-built fallback.
    NOT_APPLICABLE = "not_applicable"
    #: hard-only-v1: no review by design.
    NOT_REVIEWED_BY_DESIGN = "not_reviewed_by_design"


class ExecutionOutcome(str, Enum):
    COMPLETED = "completed"
    PROVIDER_FAILED = "provider_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    RUNTIME_FAILED = "runtime_failed"


class KnowledgeOutcome(str, Enum):
    ANSWERED_FROM_ARTIFACTS = "answered_from_artifacts"
    #: Claims with refs, and the system's review found them supported. Not
    #: scientific ground truth.
    SUPPORTED_ASSESSMENT = "supported_assessment"
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    NEEDS_USER_INPUT = "needs_user_input"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    #: The run produced no assessment to classify.
    NONE = "none"


#: Review outcomes (domain/answer_release.ReleaseOutcome values) -> status.
_REVIEW_STATUS = {
    "reviewed": ReviewStatus.PASSED,
    "repaired_and_reviewed": ReviewStatus.REPAIRED_AND_PASSED,
    "released_with_unresolved_review": ReviewStatus.BLOCKED,
    "fallback": ReviewStatus.NOT_APPLICABLE,
}

#: The policy, as data: what each review status may publish under each mode.
#: Hashed into every result, so a recorded disposition names the table that
#: produced it.
POLICY_TABLE: Mapping[str, Mapping[str, str]] = {
    ReleasePolicy.STRICT_REVIEWED_V1.value: {
        ReviewStatus.PASSED.value: Disposition.PUBLISH.value,
        ReviewStatus.REPAIRED_AND_PASSED.value: Disposition.PUBLISH.value,
        ReviewStatus.BLOCKED.value: Disposition.PARTIAL_FACTS.value,
        ReviewStatus.INCOMPLETE.value: Disposition.PARTIAL_FACTS.value,
        ReviewStatus.NOT_COMPLETED.value: Disposition.PARTIAL_FACTS.value,
        # No reviewable claim: every claim is a server-resolved value or a
        # field the hard validator checked in full, so it may be published.
        ReviewStatus.NOT_APPLICABLE.value: Disposition.PUBLISH.value,
    },
    ReleasePolicy.HARD_ONLY_V1.value: {
        ReviewStatus.NOT_REVIEWED_BY_DESIGN.value: Disposition.PUBLISH.value,
        ReviewStatus.NOT_APPLICABLE.value: Disposition.PUBLISH.value,
    },
}


def policy_sha256(policy: ReleasePolicy) -> str:
    return content_sha256({"policy": policy.value, "table": dict(POLICY_TABLE[policy.value])})


def review_status_of(outcome: str, *, ungated_reason: str | None = None) -> ReviewStatus:
    """The review status a release outcome stands for. ``ungated`` is split
    by why: a review that judged only some claims is incomplete; one with no
    reviewable claim had nothing to do; anything else did not complete."""
    if outcome in _REVIEW_STATUS:
        return _REVIEW_STATUS[outcome]
    reason = (ungated_reason or "").lower()
    if "no reviewable claim" in reason:
        return ReviewStatus.NOT_APPLICABLE
    if "did not complete" in reason or "failed" in reason or "no usable" in reason:
        return ReviewStatus.NOT_COMPLETED
    if "not judged" in reason:
        return ReviewStatus.INCOMPLETE
    return ReviewStatus.NOT_COMPLETED


@dataclass(frozen=True, slots=True)
class PublicationDecision:
    disposition: Disposition
    review_status: ReviewStatus
    reason: str


def decide_publication(
    policy: ReleasePolicy, *, review_status: ReviewStatus | None, has_candidate: bool,
    is_fallback: bool = False,
) -> PublicationDecision:
    """What the user is shown. Pure: the same inputs give the same decision.

    ``has_candidate`` false (the run ended with no answer at all) is a
    withhold whatever the review; a server-built fallback carries only facts
    the server resolved, so it is partial facts, never a published answer.
    """
    if policy is ReleasePolicy.LEGACY:
        raise ValueError("the legacy policy records no publication decision")
    if not has_candidate:
        return PublicationDecision(
            Disposition.WITHHOLD, review_status or ReviewStatus.NOT_COMPLETED,
            "the run ended without an answer candidate",
        )
    if is_fallback:
        return PublicationDecision(
            Disposition.PARTIAL_FACTS, ReviewStatus.NOT_APPLICABLE,
            "the deterministic fallback states only server-resolved facts",
        )
    status = review_status or ReviewStatus.NOT_COMPLETED
    if policy is ReleasePolicy.HARD_ONLY_V1 and status is not ReviewStatus.NOT_APPLICABLE:
        status = ReviewStatus.NOT_REVIEWED_BY_DESIGN
    disposition = Disposition(
        POLICY_TABLE[policy.value].get(status.value, Disposition.PARTIAL_FACTS.value)
    )
    reason = {
        Disposition.PUBLISH: "every reviewed claim was judged supported"
        if status in (ReviewStatus.PASSED, ReviewStatus.REPAIRED_AND_PASSED)
        else "no claim needed a semantic review" if status is ReviewStatus.NOT_APPLICABLE
        else "hard-only policy: published without semantic review, by design",
        Disposition.PARTIAL_FACTS: {
            ReviewStatus.BLOCKED: "the reviewer still blocked a claim after the one repair",
            ReviewStatus.INCOMPLETE: "the review judged only some claims",
            ReviewStatus.NOT_COMPLETED: "the review did not complete",
        }.get(status, "the answer was not cleared for publication"),
        Disposition.WITHHOLD: "nothing could be shown",
    }[disposition]
    return PublicationDecision(disposition, status, reason)


@dataclass(frozen=True, slots=True)
class RunResultV1:
    run_id: str
    session_id: str
    disposition: Disposition
    review_status: ReviewStatus
    execution_outcome: ExecutionOutcome
    knowledge_outcome: KnowledgeOutcome
    release_policy: ReleasePolicy
    created_at: datetime
    #: The answer a reader is served: the published model answer, or the
    #: fallback for partial facts. Never a withheld candidate.
    public_answer_id: str | None = None
    #: The last candidate, for the audit view only.
    candidate_answer_id: str | None = None
    #: Facts the server resolved itself (field-backed claims), the gaps, the
    #: reviews, the repair effect, the reason -- whatever the reader needs.
    detail: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_id(self.run_id, RUN, field="run_result.run_id")
        require_id(self.session_id, SESSION, field="run_result.session_id")
        for name in ("public_answer_id", "candidate_answer_id"):
            value = getattr(self, name)
            if value is not None:
                require_id(value, ANSWER, field=f"run_result.{name}")
        if self.disposition is Disposition.PUBLISH and self.public_answer_id is None:
            raise ValueError("a published result names the answer it publishes")
        if self.disposition is Disposition.WITHHOLD and self.public_answer_id is not None:
            raise ValueError("a withheld result serves no answer")
        if self.release_policy is ReleasePolicy.LEGACY:
            raise ValueError("a run result is recorded only under a declared policy")

    @property
    def policy_sha256(self) -> str:
        return policy_sha256(self.release_policy)

    @property
    def published(self) -> bool:
        return self.disposition is Disposition.PUBLISH

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "disposition": self.disposition.value,
            "review_status": self.review_status.value,
            "execution_outcome": self.execution_outcome.value,
            "knowledge_outcome": self.knowledge_outcome.value,
            "release_policy": self.release_policy.value,
            "release_policy_sha256": self.policy_sha256,
            "public_answer_id": self.public_answer_id,
            "candidate_answer_id": self.candidate_answer_id,
            "detail": dict(self.detail),
            "created_at": self.created_at.isoformat(),
        }


def knowledge_outcome_of(stop_reason: str | None, *, published: bool) -> KnowledgeOutcome:
    """The knowledge axis, from the decision state's stop reason. A supported
    assessment needs a published answer behind it."""
    if stop_reason == "answered_from_artifacts":
        return KnowledgeOutcome.ANSWERED_FROM_ARTIFACTS
    if stop_reason == "needs_user_input":
        return KnowledgeOutcome.NEEDS_USER_INPUT
    if stop_reason == "sufficient":
        return KnowledgeOutcome.SUPPORTED_ASSESSMENT if published else \
            KnowledgeOutcome.EVIDENCE_INSUFFICIENT
    if stop_reason in ("insufficient_evidence", "budget_exhausted"):
        return KnowledgeOutcome.EVIDENCE_INSUFFICIENT
    return KnowledgeOutcome.NONE


def execution_outcome_of(failure_code: str | None, *, run_status: str,
                         budget_exhausted: bool = False) -> ExecutionOutcome:
    """The execution axis. A provider that failed is never folded into
    "completed with insufficient evidence"."""
    if run_status == "cancelled":
        return ExecutionOutcome.CANCELLED
    if failure_code in ("evidence_unavailable", "provider_rate_limited", "runtime_unavailable"):
        return ExecutionOutcome.PROVIDER_FAILED
    if run_status == "completed":
        return ExecutionOutcome.BUDGET_EXHAUSTED if budget_exhausted else ExecutionOutcome.COMPLETED
    if failure_code in ("deadline_exceeded", "phase_budget_exhausted"):
        return ExecutionOutcome.BUDGET_EXHAUSTED
    return ExecutionOutcome.RUNTIME_FAILED
