"""Target-specific policy over activity estimates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .activity import ActivityApplicability, ActivitySnapshot, finite_number
from .ids import ACTIVITY_SNAPSHOT, PRIORITISATION, new_id, require_id


class PriorityStatus(str, Enum):
    PRIORITISE = "prioritise"
    REVIEW = "review"
    LOW_PRIORITY = "low_priority"


@dataclass(frozen=True, slots=True)
class PrioritisationDecision:
    id: str
    activity_snapshot_id: str
    status: PriorityStatus
    policy_id: str
    target_id: str
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
        if not self.threshold_source.strip():
            raise ValueError("threshold_source is required")
        if not self.reason_codes:
            raise ValueError("a prioritisation decision requires at least one reason code")


@dataclass(frozen=True, slots=True)
class TargetPriorityPolicy:
    """Conservative target-specific policy over an activity interval."""

    policy_id: str
    target_id: str
    threshold_pic50: float
    threshold_source: str
    max_uncertainty: float

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not self.target_id.strip():
            raise ValueError("policy_id and target_id are required")
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

    def decide(self, snapshot: ActivitySnapshot, *, now: datetime) -> PrioritisationDecision:
        if snapshot.target.target_id != self.target_id:
            raise ValueError(
                f"policy for target {self.target_id!r} cannot decide snapshot for "
                f"{snapshot.target.target_id!r}"
            )

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
            threshold_pic50=self.threshold_pic50,
            threshold_source=self.threshold_source,
            reason_codes=tuple(reasons),
            created_at=now,
        )
