"""Accept or refuse one synthesis, at the single LLM boundary of a build (PR-12).

PR-11 built the schema, the compiler and the gates as pure functions. This is
where a model's submission meets them, inside a tool call, so a refusal reaches
the model while it can still act on it.

Acceptance changes exactly one thing: the synthesis is stored on the build.
Nothing is published here. The orchestrator's validating and rendering stages
re-run the same compiler and gates on the stored synthesis and only then write
an artifact — so the rule "a report is published only if it passes" does not
depend on this call having been made by a well-behaved caller, and a build
recovered after a crash publishes from the checkpoint rather than asking a
model to write the same prose a second time.

The correction budget is the build's, stated once: one refused submission may
be followed by one more. It is counted in ``stage_state`` rather than in
``ReportBuild.correction_attempts``, because that counter is bound to the
validating→synthesizing transition of the old path, and the orchestrator does
not walk the stage pointer backwards to express "try again".
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from ...platform import metrics
from ...domain.errors import Conflict, Violation
from ...domain.events import EventType
from ...domain.report import BuildStage
from ...report.synthesis_compiler import CompiledReport, compile_report
from ...report.synthesis_validator import validate_compiled_report
from ...validation.report.synthesis_wire import ReportSynthesisV3
from .inputs import ReportInputs, load_report_inputs
from .submit_draft import ReportValidationFailed

#: Where the accepted synthesis and its bookkeeping live in ``stage_state``.
SYNTHESIS_KEY = "synthesis_v3"
SYNTHESIS_SHA_KEY = "synthesis_v3_sha256"
SYNTHESIS_ATTEMPTS_KEY = "synthesis_v3_attempts"

#: Submissions a build may make: the first, and one correction.
MAX_SYNTHESIS_SUBMISSIONS = 2


def _now() -> datetime:
    return datetime.now(timezone.utc)


def synthesis_sha256(document: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class JudgedSynthesis:
    """A synthesis compiled against a bundle and put through every gate."""

    inputs: ReportInputs
    synthesis: ReportSynthesisV3
    report: CompiledReport | None
    violations: tuple[Violation, ...]

    @property
    def ok(self) -> bool:
        return self.report is not None and not self.violations


def judge(inputs: ReportInputs, synthesis: ReportSynthesisV3) -> JudgedSynthesis:
    """Compile, then gate. Deterministic; the same inputs give the same answer."""
    compiled = compile_report(
        bundle=inputs.bundle,
        synthesis=synthesis,
        search_performed=inputs.search_performed,
    )
    if not compiled.ok:
        return JudgedSynthesis(inputs, synthesis, None, tuple(compiled.violations))
    violations = validate_compiled_report(
        compiled.report,
        bundle=inputs.bundle,
        explanations=inputs.explanations_by_id,
        situation=inputs.situation,
    )
    return JudgedSynthesis(inputs, synthesis, compiled.report, tuple(violations))


def stored_synthesis(stage_state: dict[str, Any]) -> ReportSynthesisV3 | None:
    document = stage_state.get(SYNTHESIS_KEY)
    if not isinstance(document, dict):
        return None
    return ReportSynthesisV3.model_validate(document)


@dataclass(frozen=True, slots=True)
class SynthesisAccepted:
    report_build_id: str
    synthesis_sha256: str
    attempt: int
    gap_count: int


class SubmitReportSynthesis:
    def __init__(self, database) -> None:
        self._db = database

    async def execute(
        self, *, session_id: str, run_id: str, synthesis: ReportSynthesisV3
    ) -> SynthesisAccepted:
        inputs = await load_report_inputs(
            self._db, synthesis.report_build_id, session_id=session_id
        )
        build = inputs.build
        if build.is_terminal or build.report_id is not None:
            raise Conflict(
                f"this report build is {build.stage.value} and accepts no synthesis",
                build_id=build.id,
            )
        if build.stage is not BuildStage.SYNTHESIZING:
            # The orchestrator moves the pointer here immediately before it
            # dispatches the runtime. A submission at any other stage is either
            # early — the facts it quotes are not all assembled — or late.
            raise Conflict(
                f"this report build is {build.stage.value}, not synthesizing",
                build_id=build.id,
            )
        if SYNTHESIS_KEY in build.stage_state:
            raise Conflict(
                "this build already holds an accepted synthesis; it is not rewritten",
                build_id=build.id,
            )

        attempts = int(build.stage_state.get(SYNTHESIS_ATTEMPTS_KEY) or 0) + 1
        if attempts > MAX_SYNTHESIS_SUBMISSIONS:
            raise Conflict(
                "this build has used its submission and its one correction attempt",
                build_id=build.id,
            )

        judged = judge(inputs, synthesis)
        metrics.inc(
            "toxagent_report_synthesis_submissions_total",
            attempt=str(attempts), outcome="accepted" if judged.ok else "refused",
        )
        for violation in judged.violations:
            metrics.inc("toxagent_report_synthesis_violations_total", violation_code=violation.code)
        document = synthesis.model_dump(mode="json")
        digest = synthesis_sha256(document)

        async with self._db.unit_of_work() as uow:
            current = await uow.reports.get_build(build.id, session_id=session_id)
            if current is None or SYNTHESIS_KEY in current.stage_state:
                raise Conflict("the report build changed during validation", build_id=build.id)
            state = dict(current.stage_state)
            state[SYNTHESIS_ATTEMPTS_KEY] = attempts
            if judged.ok:
                state[SYNTHESIS_KEY] = document
                state[SYNTHESIS_SHA_KEY] = digest
            await uow.reports.save_build(replace(current, stage_state=state, updated_at=_now()))
            if judged.ok:
                uow.emit(
                    session_id=session_id, type=EventType.REPORT_DRAFT_SAVED,
                    entity_type="report_build", entity_id=build.id, run_id=run_id,
                    payload={
                        "schema_version": synthesis.schema_version,
                        "content_sha256": digest,
                        "attempt": attempts,
                    },
                )
            else:
                uow.emit(
                    session_id=session_id, type=EventType.REPORT_VALIDATION_FAILED,
                    entity_type="report_build", entity_id=build.id, run_id=run_id,
                    payload={
                        "schema_version": synthesis.schema_version,
                        "attempt": attempts,
                        "violations": [v.to_dict() for v in judged.violations],
                    },
                )
            await uow.commit()

        if not judged.ok:
            remaining = MAX_SYNTHESIS_SUBMISSIONS - attempts
            raise ReportValidationFailed(
                f"the report synthesis did not pass ({len(judged.violations)} violation(s), "
                "listed in details.violations). "
                + (
                    "Correct exactly those and call submit_report_synthesis once more; "
                    "one attempt remains and there is no fallback report."
                    if remaining
                    else "No attempt remains; the build will fail without a report."
                ),
                violations=list(judged.violations),
                attempts_remaining=remaining,
            )
        return SynthesisAccepted(
            report_build_id=build.id,
            synthesis_sha256=digest,
            attempt=attempts,
            gap_count=len(judged.report.gaps) if judged.report else 0,
        )
