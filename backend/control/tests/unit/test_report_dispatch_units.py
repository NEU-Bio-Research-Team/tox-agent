"""The pieces of the orchestrated report path that need no database (PR-12, PR-16).

The end-to-end suite proves an orchestrated build publishes. These pin the
decisions inside it that would otherwise only show up as a subtly wrong report:
what the translation to an artifact keeps and refuses, how much smaller the
synthesis turn's instructions are than the builder's, and which stage failure
is allowed to become a recovery.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.support.audit_fixtures import CCO_ATTRIBUTION, load
from toxagent.application.explanation.service import extract_highlights
from toxagent.application.report.dispatch import (
    OrchestratedReportBuild,
    ReportStageFailed,
    ReportSynthesisRefused,
)
from toxagent.application.report.stages import StageCheckpoint, StageStatus
from toxagent.platform.config import Settings
from toxagent.domain.errors import RuntimeUnavailable
from toxagent.domain.report import (
    BuildStage,
    ExplanationPackage,
    ExplanationStatus,
    SubstanceProfile,
)
from toxagent.harness.prompt_budget import estimate_tokens
from toxagent.harness.report_profile import compose_report_profile
from toxagent.harness.synthesis_profile import compose_synthesis_profile
from toxagent.report.synthesis_artifact import SCHEMA_VERSION_V3, to_artifact
from toxagent.report.synthesis_compiler import compile_report
from toxagent.report.fact_bundle import assemble
from toxagent.validation.report.synthesis_wire import ReportSynthesisV3

BUILD = "rpb_" + "a" * 32
ANALYSIS = "ana_" + "b" * 32
OBSERVATION = "obs_" + "c" * 32
EXPLANATION = "xpl_" + "d" * 32
SESSION = "ses_" + "e" * 32
NOW = datetime(2026, 9, 13, tzinfo=timezone.utc)

NARRATIVE = (
    "executive_summary",
    "substance_profile",
    "explanation_and_visuals",
    "external_evidence",
    "integrated_interpretation",
    "conclusions",
    "recommendations",
)


def _bundle(**overrides):
    payload = {
        "report_build_id": BUILD,
        "analysis_id": ANALYSIS,
        "predictions": {
            "herg": {
                "probability_blocker": 0.0271,
                "label": "non_blocker",
                "threshold": 0.5,
                "threshold_source": "vendor_default",
                "model_id": "herg-chemberta-v3",
            }
        },
        "served_endpoints": ["herg"],
        "selected_endpoints": ["herg"],
        "substance": {"canonical_smiles": "CCO", "preferred_name": "ethanol"},
        "explanations": [
            ExplanationPackage(
                explanation_id=EXPLANATION,
                observation_id=OBSERVATION,
                endpoint="herg",
                task=None,
                method="integrated-gradients",
                status=ExplanationStatus.COMPLETED,
                highlights=extract_highlights(load(CCO_ATTRIBUTION)["payload"]),
            )
        ],
        "observation_ids": {"herg": OBSERVATION},
        "required_limitations": [
            "uncalibrated_probability",
            "attribution_not_causality",
            "screening_not_safety_assessment",
        ],
        "policy": {"include_external_evidence": False},
    }
    payload.update(overrides)
    return assemble(**payload)


def _compiled(bundle):
    probability = bundle.by_path()["predictions.herg.probability_blocker"].id
    synthesis = ReportSynthesisV3.model_validate(
        {
            "report_build_id": BUILD,
            "title": "hERG screening report for ethanol",
            "sections": [
                {
                    "section_id": sid,
                    "heading": sid.replace("_", " ").title(),
                    "prose_markdown": (
                        f"Probability {{{{{probability}}}}}." if sid == "executive_summary"
                        else "Narrative."
                    ),
                    "basis_fact_ids": [probability] if sid == "executive_summary" else [],
                }
                for sid in NARRATIVE
            ],
            "conclusions": [
                {
                    "local_ref": "c1",
                    "text": "Low predicted liability under this model.",
                    "basis_fact_ids": [probability],
                    "endpoint": "herg",
                }
            ],
        }
    )
    result = compile_report(bundle=bundle, synthesis=synthesis, search_performed=False)
    assert result.ok, result.violations
    return result.report


# --- the artifact translation -----------------------------------------------


def test_a_compiled_report_becomes_a_v3_artifact_under_its_own_content_hash():
    bundle = _bundle()
    report = _compiled(bundle)
    artifact = to_artifact(
        report,
        session_id=SESSION,
        subject=SubstanceProfile(canonical_smiles="CCO", preferred_name="ethanol"),
        explanations=(),
        synthesis_sha256="sha256:abc",
        now=NOW,
    )
    assert artifact.schema_version == SCHEMA_VERSION_V3
    assert artifact.content_sha256 == report.content_sha256()
    # The compiler recorded "no search was requested"; a partial report says so.
    assert artifact.status is BuildStage.COMPLETED_WITH_GAPS
    assert {gap.reason.value for gap in artifact.gaps} >= {"external_evidence_not_requested"}
    assert artifact.claims == () and artifact.tables == ()
    summary = artifact.section_by_id["executive_summary"]
    assert "{{fct_" not in summary.body_markdown
    assert [c.basis_claim_ids for c in artifact.conclusions] == [
        (bundle.by_path()["predictions.herg.probability_blocker"].id,)
    ]
    document = artifact.to_dict()
    assert document["provenance"]["synthesis_sha256"] == "sha256:abc"
    assert {item["code"] for item in document["limitations"]} >= {"screening_not_safety_assessment"}


def test_a_gap_the_schema_cannot_count_refuses_translation_rather_than_vanishing():
    bundle = _bundle(extra_gaps=[{"reason": "made_up_reason", "section_id": "x", "detail": "?"}])
    report = _compiled(bundle)
    with pytest.raises(ValueError, match="made_up_reason"):
        to_artifact(
            report,
            session_id=SESSION,
            subject=SubstanceProfile(canonical_smiles="CCO"),
            now=NOW,
        )


# --- PR-16's second half: the synthesis turn is not sent the builder's manual --


def test_the_synthesis_instructions_are_a_fraction_of_the_builder_profile():
    profiles_dir = Settings.from_env().profiles_dir
    builder = compose_report_profile(profiles_dir)
    synthesis = compose_synthesis_profile(profiles_dir)

    builder_tokens = estimate_tokens(builder.instructions)
    synthesis_tokens = estimate_tokens(synthesis.instructions)
    assert synthesis_tokens * 3 <= builder_tokens, (synthesis_tokens, builder_tokens)
    for tool in (
        "save_report_draft", "patch_saved_report_draft", "get_report_context",
        "search_toxicology_evidence", "get_or_create_explanation",
    ):
        assert tool not in synthesis.instructions
    # Shared, not copied: a wording rule fixed once is fixed for both. The
    # builder's source-hierarchy reference is not shared, because it is written
    # in terms of read tools this turn cannot see — which is what this test
    # caught the first time it ran.
    assert set(synthesis.file_hashes) == {
        "report_synthesis/AGENTS.md",
        "report_build/references/wording-and-safety-policy.md",
    }


def test_composition_is_deterministic():
    profiles_dir = Settings.from_env().profiles_dir
    assert (
        compose_synthesis_profile(profiles_dir).content_sha256
        == compose_synthesis_profile(profiles_dir).content_sha256
    )


# --- which failures may recover ---------------------------------------------


class _Context:
    report_build_id = BUILD


def _failed(stage: BuildStage, detail: str) -> StageCheckpoint:
    return StageCheckpoint(stage=stage.value, status=StageStatus.FAILED, detail=detail)


async def _failure(checkpoint):
    dispatch = OrchestratedReportBuild.__new__(OrchestratedReportBuild)
    return await dispatch._failure(_Context(), checkpoint)


@pytest.mark.anyio
async def test_a_lost_runtime_at_synthesis_is_the_one_failure_recovery_may_retry():
    error = await _failure(
        _failed(BuildStage.SYNTHESIZING, "RuntimeUnavailable: the runtime session was lost")
    )
    assert isinstance(error, RuntimeUnavailable)


@pytest.mark.anyio
async def test_a_refused_synthesis_fails_as_validation_and_is_not_retried():
    error = await _failure(
        _failed(BuildStage.SYNTHESIZING, "ReportSynthesisRefused: refused 2 time(s)")
    )
    assert isinstance(error, ReportSynthesisRefused)
    assert error.code == "report_validation_failed"


@pytest.mark.anyio
async def test_a_deterministic_stage_failure_is_a_stage_failure():
    error = await _failure(_failed(BuildStage.PREPARING_ANALYSIS, "LookupError: gone"))
    assert isinstance(error, ReportStageFailed)
    assert error.detail["stage"] == "preparing_analysis"
