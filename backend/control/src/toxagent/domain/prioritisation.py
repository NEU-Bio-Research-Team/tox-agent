"""Target-specific policy over activity estimates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .activity import (
    ActivityApplicability,
    ActivitySnapshot,
    TargetRole,
    finite_number,
)
from .ids import ACTIVITY_SNAPSHOT, PRIORITISATION, new_id, require_id


class PriorityStatus(str, Enum):
    PRIORITISE = "prioritise"
    REVIEW = "review"
    LOW_PRIORITY = "low_priority"


class PolicyNotApplicable(ValueError):
    """The policy was asked to decide outside the scope it was defined for."""


class UnsupportedTargetRole(PolicyNotApplicable):
    """No decision rule has been defined for this target role."""


@dataclass(frozen=True, slots=True)
class PrioritisationDecision:
    id: str
    activity_snapshot_id: str
    status: PriorityStatus
    policy_id: str
    target_id: str
    assay_context: str
    intended_target_role: TargetRole
    threshold_pic50: float
    threshold_source: str
    reason_codes: tuple[str, ...]
    created_at: datetime

    def __post_init__(self) -> None:
        require_id(self.id, PRIORITISATION, field="prioritisation.id")
        require_id(
            self.activity_snapshot_id,
            ACTIVITY_SNAPSHOT,
            field="prioritisation.activity_snapshot_id",
        )
        object.__setattr__(
            self,
            "threshold_pic50",
            finite_number(self.threshold_pic50, "threshold_pic50"),
        )
        if not self.policy_id.strip() or not self.target_id.strip():
            raise ValueError("policy_id and target_id are required")
        if not self.assay_context.strip():
            raise ValueError("assay_context is required: a threshold has no meaning without it")
        if not self.threshold_source.strip():
            raise ValueError("threshold_source is required")
        if not self.reason_codes:
            raise ValueError("a prioritisation decision requires at least one reason code")


@dataclass(frozen=True, slots=True)
class TargetPriorityPolicy:
    """Conservative policy over an activity interval, scoped to one context.

    The policy key is ``(target_id, assay_context)``, not ``target_id`` alone.
    A pIC50 threshold is calibrated against a particular kind of measurement,
    so applying it to a snapshot from another assay context would compare two
    different quantities.

    ``intended_target_role`` records what the threshold assumes.  The rule
    below ("the whole interval clears the threshold" means prioritise) is only
    defined for an intended target; other roles are refused rather than
    silently inverted.
    """

    policy_id: str
    target_id: str
    assay_context: str
    threshold_pic50: float
    threshold_source: str
    max_uncertainty: float
    intended_target_role: TargetRole = TargetRole.INTENDED_TARGET

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.target_id.strip():
            raise ValueError("policy_id and target_id are required")
        if not self.assay_context.strip():
            raise ValueError("assay_context is required: a threshold is calibrated per context")
        if not self.threshold_source.strip():
            raise ValueError("threshold_source is required")
        object.__setattr__(
            self,
            "threshold_pic50",
            finite_number(self.threshold_pic50, "threshold_pic50"),
        )
        maximum = finite_number(self.max_uncertainty, "max_uncertainty")
        if maximum < 0:
            raise ValueError("max_uncertainty must be non-negative")
        object.__setattr__(self, "max_uncertainty", maximum)

    def _require_in_scope(self, snapshot: ActivitySnapshot) -> None:
        if snapshot.target.target_id != self.target_id:
            raise PolicyNotApplicable(
                f"policy for target {self.target_id!r} cannot decide snapshot for "
                f"{snapshot.target.target_id!r}"
            )
        if snapshot.assay_context != self.assay_context:
            raise PolicyNotApplicable(
                f"policy threshold is calibrated for assay context "
                f"{self.assay_context!r} and cannot decide a snapshot measured in "
                f"{snapshot.assay_context!r}"
            )
        if self.intended_target_role is not TargetRole.INTENDED_TARGET:
            raise UnsupportedTargetRole(
                f"no prioritisation rule is defined for target role "
                f"{self.intended_target_role.value!r}; the interval-above-threshold "
                "rule assumes an intended target"
            )

    def decide(self, snapshot: ActivitySnapshot, *, now: datetime) -> PrioritisationDecision:
        self._require_in_scope(snapshot)

        estimate = snapshot.estimate
        reasons: list[str] = []
        if snapshot.joint_applicability is not ActivityApplicability.IN_DOMAIN:
            status = PriorityStatus.REVIEW
            reasons.append(f"joint_applicability_{snapshot.joint_applicability.value}")
        elif estimate.uncertainty is None:
            status = PriorityStatus.REVIEW
            reasons.append("uncertainty_unavailable")
        elif estimate.uncertainty > self.max_uncertainty:
            status = PriorityStatus.REVIEW
            reasons.append("uncertainty_above_policy_limit")
        elif not estimate.has_interval:
            status = PriorityStatus.REVIEW
            reasons.append("prediction_interval_unavailable")
        elif estimate.interval_low is not None and estimate.interval_low >= self.threshold_pic50:
            status = PriorityStatus.PRIORITISE
            reasons.extend(("interval_above_target_threshold", "in_domain"))
        elif estimate.interval_high is not None and estimate.interval_high < self.threshold_pic50:
            status = PriorityStatus.LOW_PRIORITY
            reasons.extend(("interval_below_target_threshold", "in_domain"))
        else:
            status = PriorityStatus.REVIEW
            reasons.append("prediction_interval_crosses_target_threshold")

        return PrioritisationDecision(
            id=new_id(PRIORITISATION),
            activity_snapshot_id=snapshot.id,
            status=status,
            policy_id=self.policy_id,
            target_id=self.target_id,
            assay_context=self.assay_context,
            intended_target_role=self.intended_target_role,
            threshold_pic50=self.threshold_pic50,
            threshold_source=self.threshold_source,
            reason_codes=tuple(reasons),
            created_at=now,
        )
