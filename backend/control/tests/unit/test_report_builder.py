from datetime import datetime, timezone

import pytest

from toxagent.platform.config import PACKAGE_ROOT
from toxagent.domain.ids import new_id
from toxagent.domain.report import (
    BuildStage,
    BuildTransitionError,
    ReportBuild,
    ReportBuildRequest,
    REQUIRED_SECTION_IDS,
)
from toxagent.harness.report_profile import compose_report_profile
from toxagent.validation.report.draft_wire import ReportDraftCandidate


NOW = datetime(2026, 9, 9, tzinfo=timezone.utc)


def test_report_profile_composition_is_complete_and_deterministic():
    first = compose_report_profile(PACKAGE_ROOT / "agent_profiles")
    second = compose_report_profile(PACKAGE_ROOT / "agent_profiles")

    assert first.skills == (
        "assemble-report-context",
        "explain-predictor-results",
        "research-toxicology-evidence",
        "compose-scientific-report",
        "preflight-report-draft",
    )
    assert len(first.file_hashes) == 15
    assert first.instructions == second.instructions
    assert first.content_sha256 == second.content_sha256
    assert "submit_report_draft" in first.instructions


def test_report_build_correction_is_bounded_and_persistable():
    request = ReportBuildRequest(
        session_id=new_id("ses"), analysis_id=new_id("ana"),
        selected_endpoints=("herg",),
    )
    build = ReportBuild.start(
        session_id=request.session_id, run_id=new_id("run"), request=request, now=NOW
    )
    for stage in (
        BuildStage.PREPARING_ANALYSIS, BuildStage.ASSEMBLING_SUBSTANCE,
        BuildStage.ASSEMBLING_PREDICTIONS, BuildStage.SYNTHESIZING,
        BuildStage.VALIDATING, BuildStage.SYNTHESIZING,
    ):
        build = build.advance(stage, now=NOW)
    assert build.correction_attempts == 1
    build = build.advance(BuildStage.VALIDATING, now=NOW)
    with pytest.raises(BuildTransitionError):
        build.advance(BuildStage.SYNTHESIZING, now=NOW)


def test_report_draft_requires_every_stable_section_id():
    sections = [
        {
            "section_id": section_id,
            "heading": section_id,
            "body_markdown": "Recorded content.",
        }
        for section_id in REQUIRED_SECTION_IDS
    ]
    draft = ReportDraftCandidate.model_validate({
        "report_build_id": new_id("rpb"),
        "title": "Toxicity Screening Report",
        "sections": sections,
    })
    assert tuple(section.section_id for section in draft.sections) == REQUIRED_SECTION_IDS

    with pytest.raises(ValueError):
        ReportDraftCandidate.model_validate({
            "report_build_id": new_id("rpb"),
            "title": "Incomplete",
            "sections": sections[:-1],
        })


# --- why no report exists ----------------------------------------------------


def _failed_build() -> ReportBuild:
    request = ReportBuildRequest(
        session_id=new_id("ses"), analysis_id=new_id("ana"), selected_endpoints=("herg",),
    )
    build = ReportBuild.start(
        session_id=request.session_id, run_id=new_id("run"), request=request, now=NOW
    )
    for stage in (
        BuildStage.PREPARING_ANALYSIS, BuildStage.ASSEMBLING_SUBSTANCE,
        BuildStage.ASSEMBLING_PREDICTIONS, BuildStage.SYNTHESIZING, BuildStage.VALIDATING,
    ):
        build = build.advance(stage, now=NOW)
    return build.advance(
        BuildStage.FAILED, now=NOW,
        failure_code="report_validation_failed",
        failure_detail="2 violation(s) survived the one permitted correction attempt",
    )


def test_a_refused_draft_is_not_reported_as_a_tool_that_was_never_called():
    """A live run failed with "the runtime reached a terminal event without
    submit_report_draft" after calling that tool twice and being refused twice.

    The message was emitted for every reason a report might be absent, so it
    sent whoever read it looking for a runtime that skipped a tool call, while
    the event log showed two calls and two rejections. The distinction is the
    difference between debugging the runtime and debugging the draft.
    """
    from toxagent.harness.gateway import AgentRuntimeGateway

    detail = AgentRuntimeGateway._report_failure_detail(object(), _failed_build())
    assert "submit_report_draft was called" in detail
    assert "did not pass validation" in detail
    assert "survived the one permitted correction attempt" in detail


def test_a_build_that_never_reached_a_submission_says_which_stage_it_stalled_in():
    from toxagent.harness.gateway import AgentRuntimeGateway

    request = ReportBuildRequest(
        session_id=new_id("ses"), analysis_id=new_id("ana"), selected_endpoints=("herg",),
    )
    build = ReportBuild.start(
        session_id=request.session_id, run_id=new_id("run"), request=request, now=NOW
    ).advance(BuildStage.PREPARING_ANALYSIS, now=NOW)

    detail = AgentRuntimeGateway._report_failure_detail(object(), build)
    assert "preparing_analysis" in detail
    assert "never produced an artifact" in detail


def test_a_vanished_run_and_a_vanished_build_are_distinguished():
    from toxagent.harness.gateway import AgentRuntimeGateway

    assert "run disappeared" in AgentRuntimeGateway._report_failure_detail(None, _failed_build())
    assert "no longer resolves" in AgentRuntimeGateway._report_failure_detail(object(), None)
