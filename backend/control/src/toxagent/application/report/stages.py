"""Typed stage checkpoints for a report build (WS05 / PR-09).

The audit's report build walked through five stages at one timestamp before the
runtime was ever dispatched, then emitted a single ``report.stage_changed``
saying "synthesizing". The stage history was fiction: nothing had been prepared,
no substance assembled, no predictions projected. A user watching the build saw
one event and then 155 seconds of silence.

A stage record has to answer three questions a recovering worker actually asks:

* **did this stage run, and did it finish?** ``status`` — running, completed,
  skipped or failed — not merely "the build's stage pointer moved past it";
* **what did it produce?** ``output_refs``, so a resumed build reuses the
  explanation it already paid a predictor for instead of computing it again;
* **how many times has it been tried?** ``attempt``, so a stage that fails
  twice is visible as a stage that failed twice.

``skipped`` is a first-class outcome and deliberately not the same as absent. A
build with ``include_external_evidence=false`` did not "not reach" the evidence
stage — it reached it and correctly did nothing, and that distinction is what
keeps a report from later claiming a search happened (P0-2).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from ...domain.report import BuildStage, ReportBuildRequest

#: Bumped when the checkpoint shape changes. A reader that does not recognise a
#: version treats the checkpoint as absent and re-runs the stage, which is safe
#: for every deterministic stage and is why the expensive ones carry refs.
CHECKPOINT_SCHEMA_VERSION = "stage-checkpoint-1"

#: Where checkpoints live inside ``ReportBuild.stage_state``. Namespaced so the
#: existing keys in that dict keep working untouched.
CHECKPOINT_KEY = "stage_checkpoints"


class StageStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    #: Reached, correctly did nothing. Not the same as never reached.
    SKIPPED = "skipped"
    FAILED = "failed"

    @property
    def is_settled(self) -> bool:
        return self in (StageStatus.COMPLETED, StageStatus.SKIPPED)


def output_hash(output: Any) -> str:
    """A stable digest of a stage's output, for the audit trail.

    Sorted keys and a string fallback: this has to be reproducible across
    processes, and a dict whose iteration order changed would otherwise look
    like a stage that produced something different.
    """
    return "sha256:" + hashlib.sha256(
        json.dumps(output, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class StageCheckpoint:
    stage: str
    status: StageStatus
    attempt: int = 1
    started_at: str | None = None
    completed_at: str | None = None
    #: Ids of what this stage produced — explanation ids, evidence ids, the
    #: substance profile ref. Never the payloads: those live where they live,
    #: and copying them here would give an artifact two sources.
    output_refs: dict[str, Any] = field(default_factory=dict)
    output_hash: str = ""
    #: Why a stage did nothing, or why it failed. Required for both, because
    #: "skipped" with no reason is indistinguishable from a bug.
    detail: str = ""
    schema_version: str = CHECKPOINT_SCHEMA_VERSION

    @property
    def is_reusable(self) -> bool:
        """Whether a recovering build may take this stage's word for it."""
        return (
            self.schema_version == CHECKPOINT_SCHEMA_VERSION
            and self.status.is_settled
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status.value,
            "attempt": self.attempt,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "output_refs": dict(self.output_refs),
            "output_hash": self.output_hash,
            "detail": self.detail,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageCheckpoint":
        try:
            status = StageStatus(payload.get("status"))
        except ValueError:
            status = StageStatus.FAILED
        return cls(
            stage=str(payload.get("stage") or ""),
            status=status,
            attempt=int(payload.get("attempt") or 1),
            started_at=payload.get("started_at"),
            completed_at=payload.get("completed_at"),
            output_refs=dict(payload.get("output_refs") or {}),
            output_hash=str(payload.get("output_hash") or ""),
            detail=str(payload.get("detail") or ""),
            schema_version=str(payload.get("schema_version") or ""),
        )

    def started(self, *, now: datetime, attempt: int) -> "StageCheckpoint":
        return replace(
            self,
            status=StageStatus.RUNNING,
            attempt=attempt,
            started_at=now.isoformat(),
            completed_at=None,
            schema_version=CHECKPOINT_SCHEMA_VERSION,
        )

    def completed(self, *, now: datetime, output: Any) -> "StageCheckpoint":
        refs = dict(output) if isinstance(output, Mapping) else {"value": output}
        return replace(
            self,
            status=StageStatus.COMPLETED,
            completed_at=now.isoformat(),
            output_refs=refs,
            output_hash=output_hash(refs),
        )

    def skipped(self, *, now: datetime, reason: str) -> "StageCheckpoint":
        return replace(
            self,
            status=StageStatus.SKIPPED,
            completed_at=now.isoformat(),
            detail=reason,
            output_refs={},
            output_hash="",
        )

    def failed(self, *, now: datetime, reason: str) -> "StageCheckpoint":
        return replace(
            self,
            status=StageStatus.FAILED,
            completed_at=now.isoformat(),
            detail=reason,
        )


def read_checkpoints(stage_state: Mapping[str, Any]) -> dict[str, StageCheckpoint]:
    raw = stage_state.get(CHECKPOINT_KEY)
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(stage): StageCheckpoint.from_dict(payload)
        for stage, payload in raw.items()
        if isinstance(payload, Mapping)
    }


def write_checkpoint(
    stage_state: Mapping[str, Any], checkpoint: StageCheckpoint
) -> dict[str, Any]:
    """A new ``stage_state`` with this checkpoint stored. Never mutates."""
    state = dict(stage_state)
    existing = dict(state.get(CHECKPOINT_KEY) or {})
    existing[checkpoint.stage] = checkpoint.to_dict()
    state[CHECKPOINT_KEY] = existing
    return state


#: The deterministic assembly stages, in order, before the model is involved.
#: Everything here is server work; the only LLM boundary in a build is
#: SYNTHESIZING.
ASSEMBLY_STAGES: tuple[BuildStage, ...] = (
    BuildStage.PREPARING_ANALYSIS,
    BuildStage.ASSEMBLING_SUBSTANCE,
    BuildStage.ASSEMBLING_PREDICTIONS,
    BuildStage.GENERATING_EXPLANATIONS,
    BuildStage.RESEARCHING_EVIDENCE,
)

#: The whole forward path a build walks, assembly then synthesis then output.
PIPELINE: tuple[BuildStage, ...] = ASSEMBLY_STAGES + (
    BuildStage.SYNTHESIZING,
    BuildStage.VALIDATING,
    BuildStage.RENDERING,
)

#: Stages whose work is skippable by request, and what to record when it is.
_OPTIONAL: Mapping[BuildStage, tuple[str, str]] = {
    BuildStage.GENERATING_EXPLANATIONS: (
        "include_explanations",
        "this build did not ask for explanations",
    ),
    BuildStage.RESEARCHING_EVIDENCE: (
        "include_external_evidence",
        "this build did not ask for external evidence",
    ),
}


@dataclass(frozen=True, slots=True)
class PlannedStage:
    stage: BuildStage
    #: False when the request switched this stage off. The stage is still
    #: walked and still checkpointed — as ``skipped``, with this reason.
    will_run: bool = True
    skip_reason: str = ""


def plan_stages(request: ReportBuildRequest) -> tuple[PlannedStage, ...]:
    """Every stage this build will walk, and which of them will do work.

    The whole pipeline is always returned. An optional stage that was switched
    off appears with ``will_run=False`` rather than being dropped, because a
    report has to be able to say "no search was performed" and "the search
    found nothing" as different things (P0-2).
    """
    planned: list[PlannedStage] = []
    for stage in PIPELINE:
        optional = _OPTIONAL.get(stage)
        if optional is None:
            planned.append(PlannedStage(stage))
            continue
        attribute, reason = optional
        if getattr(request, attribute, True):
            planned.append(PlannedStage(stage))
        else:
            planned.append(PlannedStage(stage, will_run=False, skip_reason=reason))
    return tuple(planned)


def remaining_stages(
    planned: Sequence[PlannedStage], checkpoints: Mapping[str, StageCheckpoint]
) -> tuple[PlannedStage, ...]:
    """The stages a resumed build still has to do.

    A stage with a reusable checkpoint is not repeated: that is the difference
    between recovering a 155-second build and paying for it twice. A stage that
    failed, or whose checkpoint was written by a schema this build does not
    recognise, is retried — every stage here is either deterministic or carries
    refs to what it already produced.
    """
    return tuple(
        item
        for item in planned
        if not (
            (checkpoint := checkpoints.get(item.stage.value)) is not None
            and checkpoint.is_reusable
        )
    )


def reusable_outputs(
    checkpoints: Mapping[str, StageCheckpoint]
) -> dict[str, dict[str, Any]]:
    """What earlier stages already produced, keyed by stage."""
    return {
        stage: dict(checkpoint.output_refs)
        for stage, checkpoint in checkpoints.items()
        if checkpoint.is_reusable and checkpoint.output_refs
    }


def stage_progress(
    planned: Sequence[PlannedStage], checkpoints: Mapping[str, StageCheckpoint]
) -> dict[str, Any]:
    """What a progress UI needs, without any prose a model wrote (WS09).

    Semantic codes and counts only. The presentation layer localises them; the
    one thing this must never carry is a sentence the model produced, because
    an ungrounded sentence in a progress feed is still an ungrounded sentence.
    """
    settled = [
        item
        for item in planned
        if (cp := checkpoints.get(item.stage.value)) is not None and cp.status.is_settled
    ]
    running = next(
        (
            item
            for item in planned
            if (cp := checkpoints.get(item.stage.value)) is not None
            and cp.status is StageStatus.RUNNING
        ),
        None,
    )
    return {
        "total": len(planned),
        "completed": len(settled),
        "current_stage": running.stage.value if running else None,
        "stages": [
            {
                "stage": item.stage.value,
                "status": (
                    checkpoints[item.stage.value].status.value
                    if item.stage.value in checkpoints
                    else "pending"
                ),
                "attempt": (
                    checkpoints[item.stage.value].attempt
                    if item.stage.value in checkpoints
                    else 0
                ),
                "will_run": item.will_run,
            }
            for item in planned
        ],
    }


def gaps_from_checkpoints(
    checkpoints: Iterable[StageCheckpoint],
) -> tuple[dict[str, str], ...]:
    """Typed gaps for every stage that did not do its work.

    A skipped evidence stage becomes ``external_evidence_not_requested``; a
    failed one becomes ``provider_unavailable``. Derived from the checkpoints
    rather than from what a model remembers, so a build cannot finish claiming
    a search it never ran.
    """
    reasons = {
        (BuildStage.RESEARCHING_EVIDENCE.value, StageStatus.SKIPPED):
            "external_evidence_not_requested",
        (BuildStage.RESEARCHING_EVIDENCE.value, StageStatus.FAILED):
            "provider_unavailable",
        (BuildStage.GENERATING_EXPLANATIONS.value, StageStatus.SKIPPED):
            "explanation_failed",
        (BuildStage.GENERATING_EXPLANATIONS.value, StageStatus.FAILED):
            "explanation_failed",
        (BuildStage.ASSEMBLING_SUBSTANCE.value, StageStatus.FAILED):
            "compound_identity_unresolved",
    }
    out: list[dict[str, str]] = []
    for checkpoint in checkpoints:
        reason = reasons.get((checkpoint.stage, checkpoint.status))
        if reason is None:
            continue
        out.append(
            {
                "reason": reason,
                "stage": checkpoint.stage,
                "detail": checkpoint.detail or f"{checkpoint.stage} did no work",
            }
        )
    return tuple(out)
