"""A report build: its stages, the allowed transitions between them, and the request that starts it."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Final

from ..ids import (
    ANALYSIS,
    REPORT,
    REPORT_BUILD,
    RUN,
    SESSION,
    new_id,
    require_id,
)
from .vocabulary import BUILD_REQUEST_SCHEMA_VERSION, SUPPORTED_REPORT_LANGUAGES


class BuildStage(str, Enum):
    """Spec section 10. Persisted so a build resumes rather than repeating a
    billable predictor or provider call after a restart."""

    QUEUED = "queued"
    PREPARING_ANALYSIS = "preparing_analysis"
    ASSEMBLING_SUBSTANCE = "assembling_substance"
    ASSEMBLING_PREDICTIONS = "assembling_predictions"
    GENERATING_EXPLANATIONS = "generating_explanations"
    RESEARCHING_EVIDENCE = "researching_evidence"
    SYNTHESIZING = "synthesizing"
    VALIDATING = "validating"
    RENDERING = "rendering"
    COMPLETED = "completed"
    COMPLETED_WITH_GAPS = "completed_with_gaps"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STAGES: Final[frozenset[BuildStage]] = frozenset(
    {
        BuildStage.COMPLETED,
        BuildStage.COMPLETED_WITH_GAPS,
        BuildStage.FAILED,
        BuildStage.CANCELLED,
    }
)


#: The forward path. Every non-terminal stage may additionally fail or be
#: cancelled; that is added below rather than repeated eleven times.
_FORWARD: Final[dict[BuildStage, frozenset[BuildStage]]] = {
    BuildStage.QUEUED: frozenset({BuildStage.PREPARING_ANALYSIS}),
    BuildStage.PREPARING_ANALYSIS: frozenset({BuildStage.ASSEMBLING_SUBSTANCE}),
    BuildStage.ASSEMBLING_SUBSTANCE: frozenset({BuildStage.ASSEMBLING_PREDICTIONS}),
    BuildStage.ASSEMBLING_PREDICTIONS: frozenset(
        # Explanations are skippable by request (include_explanations=false);
        # nothing else on this path is.
        {BuildStage.GENERATING_EXPLANATIONS, BuildStage.RESEARCHING_EVIDENCE,
         BuildStage.SYNTHESIZING}
    ),
    BuildStage.GENERATING_EXPLANATIONS: frozenset(
        {BuildStage.RESEARCHING_EVIDENCE, BuildStage.SYNTHESIZING}
    ),
    BuildStage.RESEARCHING_EVIDENCE: frozenset({BuildStage.SYNTHESIZING}),
    # Validation returns to synthesis for the one permitted correction attempt
    # (spec section 10 stage 4); the attempt cap lives on the aggregate, not
    # in this table, so the transition itself stays legal exactly once.
    BuildStage.SYNTHESIZING: frozenset({BuildStage.VALIDATING}),
    BuildStage.VALIDATING: frozenset({BuildStage.SYNTHESIZING, BuildStage.RENDERING}),
    BuildStage.RENDERING: frozenset(
        {BuildStage.COMPLETED, BuildStage.COMPLETED_WITH_GAPS}
    ),
}


ALLOWED_STAGE_TRANSITIONS: Final[dict[BuildStage, frozenset[BuildStage]]] = {
    stage: (
        frozenset()
        if stage in TERMINAL_STAGES
        else _FORWARD.get(stage, frozenset()) | {BuildStage.FAILED, BuildStage.CANCELLED}
    )
    for stage in BuildStage
}


@dataclass(frozen=True, slots=True)
class ReportBuildRequest:
    """Spec section 5.1. Validated at admission, then frozen into the build.

    The request is stored rather than re-derived because "which endpoints did
    the user actually ask for" is not recoverable from the finished report: a
    report with two endpoints could be a two-endpoint request that succeeded or
    a three-endpoint request that lost one, and those are different documents.
    """

    session_id: str
    analysis_id: str
    selected_endpoints: tuple[str, ...]
    selected_tox21_tasks: tuple[str, ...] = ()
    report_language: str = "en"
    audience: str = "technical_r_and_d"
    include_explanations: bool = True
    include_external_evidence: bool = True
    output_formats: tuple[str, ...] = ("markdown", "html")
    schema_version: str = BUILD_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_id(self.session_id, SESSION, field="report_build_request.session_id")
        require_id(self.analysis_id, ANALYSIS, field="report_build_request.analysis_id")
        if self.report_language not in SUPPORTED_REPORT_LANGUAGES:
            raise ValueError(
                f"report_language {self.report_language!r} is not served; this version is "
                f"English-first ({sorted(SUPPORTED_REPORT_LANGUAGES)})"
            )
        if not self.selected_endpoints:
            raise ValueError("a report must name at least one endpoint")
        if len(set(self.selected_endpoints)) != len(self.selected_endpoints):
            raise ValueError("selected_endpoints contains a duplicate")
        if self.selected_tox21_tasks and "tox21" not in self.selected_endpoints:
            raise ValueError("selected_tox21_tasks named without selecting the tox21 endpoint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "analysis_id": self.analysis_id,
            "report_language": self.report_language,
            "audience": self.audience,
            "selected_endpoints": list(self.selected_endpoints),
            "selected_tox21_tasks": list(self.selected_tox21_tasks),
            "include_explanations": self.include_explanations,
            "include_external_evidence": self.include_external_evidence,
            "output_formats": list(self.output_formats),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReportBuildRequest":
        return cls(
            session_id=payload["session_id"],
            analysis_id=payload["analysis_id"],
            selected_endpoints=tuple(payload.get("selected_endpoints", ())),
            selected_tox21_tasks=tuple(payload.get("selected_tox21_tasks", ())),
            report_language=payload.get("report_language", "en"),
            audience=payload.get("audience", "technical_r_and_d"),
            include_explanations=bool(payload.get("include_explanations", True)),
            include_external_evidence=bool(payload.get("include_external_evidence", True)),
            output_formats=tuple(payload.get("output_formats", ("markdown", "html"))),
        )


class BuildTransitionError(ValueError):
    """An illegal stage move. Named so a caller can distinguish "this build
    cannot go there" from "this build is malformed"."""


@dataclass(frozen=True, slots=True)
class ReportBuild:
    """The resumable aggregate (spec section 10).

    Frozen, and every transition returns a new value, so a stage change is a
    thing that gets written rather than a field that gets mutated somewhere and
    maybe persisted. ``correction_attempts`` lives here rather than in the
    validator because the cap is a property of the build, and a validator that
    counted its own retries would reset on every restart.
    """

    id: str
    session_id: str
    run_id: str
    analysis_id: str
    request: ReportBuildRequest
    stage: BuildStage
    created_at: datetime
    updated_at: datetime
    deadline_at: datetime | None = None
    report_id: str | None = None
    correction_attempts: int = 0
    failure_code: str | None = None
    failure_detail: str | None = None
    #: Stage outputs already produced, so a resumed build skips the expensive
    #: work it already paid for: explanation ids by "endpoint/task", the
    #: substance profile, the accepted evidence it read.
    stage_state: dict[str, Any] = field(default_factory=dict)

    #: Spec section 10 stage 4: one correction attempt, then stop.
    MAX_CORRECTION_ATTEMPTS = 1

    def __post_init__(self) -> None:
        require_id(self.id, REPORT_BUILD, field="report_build.id")
        require_id(self.session_id, SESSION, field="report_build.session_id")
        require_id(self.run_id, RUN, field="report_build.run_id")
        require_id(self.analysis_id, ANALYSIS, field="report_build.analysis_id")
        if self.report_id is not None:
            require_id(self.report_id, REPORT, field="report_build.report_id")

    @classmethod
    def start(
        cls,
        *,
        session_id: str,
        run_id: str,
        request: ReportBuildRequest,
        now: datetime,
        deadline_at: datetime | None = None,
    ) -> "ReportBuild":
        return cls(
            id=new_id(REPORT_BUILD),
            session_id=session_id,
            run_id=run_id,
            analysis_id=request.analysis_id,
            request=request,
            stage=BuildStage.QUEUED,
            created_at=now,
            updated_at=now,
            deadline_at=deadline_at,
        )

    @property
    def is_terminal(self) -> bool:
        return self.stage in TERMINAL_STAGES

    @property
    def corrections_exhausted(self) -> bool:
        return self.correction_attempts >= self.MAX_CORRECTION_ATTEMPTS

    def advance(self, stage: BuildStage, *, now: datetime, **changes: Any) -> "ReportBuild":
        from dataclasses import replace

        allowed = ALLOWED_STAGE_TRANSITIONS[self.stage]
        if stage not in allowed:
            raise BuildTransitionError(
                f"a report build cannot move from {self.stage.value} to {stage.value}"
            )
        if stage is BuildStage.SYNTHESIZING and self.stage is BuildStage.VALIDATING:
            # The correction loop. Counted here so the cap survives a restart
            # and a second failed draft cannot buy a third attempt.
            if self.corrections_exhausted:
                raise BuildTransitionError(
                    "this build has already used its one correction attempt; an invalid "
                    "draft is never accepted, so the build fails instead"
                )
            changes.setdefault("correction_attempts", self.correction_attempts + 1)
        return replace(self, stage=stage, updated_at=now, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_build_id": self.id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "analysis_id": self.analysis_id,
            "request": self.request.to_dict(),
            "stage": self.stage.value,
            "report_id": self.report_id,
            "correction_attempts": self.correction_attempts,
            "failure_code": self.failure_code,
            "failure_detail": self.failure_detail,
            "deadline_at": self.deadline_at.isoformat() if self.deadline_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
