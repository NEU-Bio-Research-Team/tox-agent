"""The deterministic stages, driven by the real orchestrator (PR-10).

Everything the audit's build asked a model to arrange — resolve the snapshot,
resolve the substance, project the predictions, ensure the explanations,
search the literature — done by the server, checkpointed, and skippable with a
reason.

The ports are lambdas here. That is the point: whether the evidence stage
declines correctly when the build did not ask for evidence should be a test
that runs in a millisecond, not one that needs a database and a network double.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from toxagent.application.report_orchestrator import ReportOrchestrator, StageSkipped
from toxagent.application.report_stage_handlers import (
    DeterministicHandlers,
    bundle_from_checkpoints,
)
from toxagent.application.report_stages import StageStatus, read_checkpoints
from toxagent.domain.report import (
    BuildStage,
    ExplanationHighlights,
    ExplanationPackage,
    ExplanationStatus,
    ReportBuild,
    ReportBuildRequest,
)

pytestmark = pytest.mark.anyio

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
SESSION = "ses_" + "1" * 32
RUN = "run_" + "2" * 32
ANALYSIS = "ana_" + "3" * 32
OBSERVATION = "obs_" + "4" * 32

SNAPSHOT = SimpleNamespace(
    id=ANALYSIS,
    canonical_smiles="CCO",
    served_endpoints=("herg",),
    content_sha256="sha256:abc",
)

PREDICTIONS = {
    "herg": {
        "probability_blocker": 0.0271,
        "label": "non_blocker",
        "model_id": "herg-chemberta-v3",
    }
}


def _request(**overrides) -> ReportBuildRequest:
    payload = {
        "session_id": SESSION,
        "analysis_id": ANALYSIS,
        "selected_endpoints": ("herg",),
        "include_explanations": True,
        "include_external_evidence": True,
    }
    payload.update(overrides)
    return ReportBuildRequest(**payload)


def _build(**overrides) -> ReportBuild:
    return ReportBuild.start(
        session_id=SESSION,
        run_id=RUN,
        request=overrides.pop("request", _request()),
        now=NOW,
        deadline_at=NOW + timedelta(hours=1),
    )


class FakeStore:
    def __init__(self, build: ReportBuild) -> None:
        self.build = build

    async def load(self, build_id: str):
        return self.build if build_id == self.build.id else None

    async def save(self, build: ReportBuild) -> None:
        self.build = build


class FakeEvents:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, build, event: str, payload: dict) -> None:
        self.events.append((event, payload))


def _package(endpoint="herg", task=None, status=ExplanationStatus.COMPLETED):
    return ExplanationPackage(
        explanation_id="xpl_" + "5" * 32,
        observation_id=OBSERVATION,
        endpoint=endpoint,
        task=task,
        method="integrated-gradients",
        status=status,
        highlights=ExplanationHighlights(
            negative_contributors=({"atom_index": 0},), unmapped_importance=0.2
        ),
    )


async def _analyses(analysis_id: str, *, session_id: str):
    return SNAPSHOT if analysis_id == ANALYSIS else None


async def _explanations(*, analysis_id: str, endpoint: str, task: str | None):
    return _package(endpoint, task)


async def _substances(*, canonical_smiles: str):
    return {"canonical_smiles": canonical_smiles, "preferred_name": "ethanol"}


async def _evidence(*, analysis_id: str, endpoint: str, compound_names, limit: int):
    return [{"evidence_id": "evd_" + "6" * 32, "relevance": "direct"}]


async def _run(build: ReportBuild, **ports):
    handlers = DeterministicHandlers(
        analyses=ports.get("analyses", _analyses),
        explanations=ports.get("explanations", _explanations),
        substances=ports.get("substances", _substances),
        evidence=ports.get("evidence", _evidence),
    ).as_mapping()
    store, events = FakeStore(build), FakeEvents()
    orchestrator = ReportOrchestrator(
        store=store, handlers=handlers, events=events, clock=lambda: NOW
    )
    result = await orchestrator.run(build.id)
    return result, store, events


def _checkpoint(store: FakeStore, stage: BuildStage):
    return read_checkpoints(store.build.stage_state)[stage.value]


# --- the assembly path ------------------------------------------------------


async def test_every_assembly_stage_completes_and_checkpoints_its_refs() -> None:
    _, store, _ = await _run(_build())
    for stage in (
        BuildStage.PREPARING_ANALYSIS,
        BuildStage.ASSEMBLING_SUBSTANCE,
        BuildStage.ASSEMBLING_PREDICTIONS,
        BuildStage.GENERATING_EXPLANATIONS,
        BuildStage.RESEARCHING_EVIDENCE,
    ):
        assert _checkpoint(store, stage).status is StageStatus.COMPLETED


async def test_preparing_pins_the_snapshot_and_names_the_unserved_endpoints() -> None:
    build = _build(request=_request(selected_endpoints=("herg", "clintox")))
    _, store, _ = await _run(build)
    refs = _checkpoint(store, BuildStage.PREPARING_ANALYSIS).output_refs
    assert refs["analysis_id"] == ANALYSIS
    assert refs["canonical_smiles"] == "CCO"
    assert refs["unavailable_endpoints"] == ["clintox"]


async def test_a_checkpoint_carries_refs_not_payloads() -> None:
    _, store, _ = await _run(_build())
    refs = _checkpoint(store, BuildStage.GENERATING_EXPLANATIONS).output_refs
    assert refs["explanations"] == {"herg": "xpl_" + "5" * 32}
    assert "highlights" not in refs and "atoms" not in refs


async def test_a_later_stage_reads_the_earlier_stages_refs() -> None:
    seen: dict[str, Any] = {}

    async def recording_evidence(*, analysis_id, endpoint, compound_names, limit):
        seen["names"] = list(compound_names)
        seen["endpoint"] = endpoint
        return []

    await _run(_build(), evidence=recording_evidence)
    assert seen["names"] == ["ethanol"]
    assert seen["endpoint"] == "herg"


async def test_a_missing_analysis_fails_the_stage_rather_than_inventing_one() -> None:
    async def missing(analysis_id: str, *, session_id: str):
        return None

    _, store, _ = await _run(_build(), analyses=missing)
    checkpoint = _checkpoint(store, BuildStage.PREPARING_ANALYSIS)
    assert checkpoint.status is StageStatus.FAILED
    assert "does not exist" in checkpoint.detail


# --- declining is not failing -----------------------------------------------


async def test_a_build_that_asked_for_no_evidence_skips_with_a_reason() -> None:
    build = _build(request=_request(include_external_evidence=False))
    result, store, _ = await _run(build)
    checkpoint = _checkpoint(store, BuildStage.RESEARCHING_EVIDENCE)
    assert checkpoint.status is StageStatus.SKIPPED
    assert "did not ask for external evidence" in checkpoint.detail
    assert {gap["reason"] for gap in result.gaps} == {"external_evidence_not_requested"}


async def test_no_provider_configured_is_skipped_not_failed() -> None:
    result, store, _ = await _run(_build(), evidence=None)
    checkpoint = _checkpoint(store, BuildStage.RESEARCHING_EVIDENCE)
    assert checkpoint.status is StageStatus.SKIPPED
    assert "no literature provider" in checkpoint.detail


async def test_a_provider_outage_is_a_gap_not_a_failed_report() -> None:
    async def down(*, analysis_id, endpoint, compound_names, limit):
        raise ConnectionError("provider down")

    result, store, _ = await _run(_build(), evidence=down)
    checkpoint = _checkpoint(store, BuildStage.RESEARCHING_EVIDENCE)
    assert checkpoint.status is StageStatus.SKIPPED
    assert "could not be reached" in checkpoint.detail


async def test_an_unresolvable_compound_declines_with_its_reason() -> None:
    async def unknown(*, canonical_smiles: str):
        return None

    _, store, _ = await _run(_build(), substances=unknown)
    checkpoint = _checkpoint(store, BuildStage.ASSEMBLING_SUBSTANCE)
    assert checkpoint.status is StageStatus.SKIPPED
    assert "no record for this structure" in checkpoint.detail


async def test_a_compound_provider_outage_declines_rather_than_failing() -> None:
    async def down(*, canonical_smiles: str):
        raise TimeoutError("pubchem")

    _, store, _ = await _run(_build(), substances=down)
    assert _checkpoint(store, BuildStage.ASSEMBLING_SUBSTANCE).status is StageStatus.SKIPPED


async def test_explanations_switched_off_are_skipped() -> None:
    build = _build(request=_request(include_explanations=False))
    _, store, _ = await _run(build)
    assert _checkpoint(store, BuildStage.GENERATING_EXPLANATIONS).status is StageStatus.SKIPPED


async def test_one_failed_explanation_target_does_not_lose_the_others() -> None:
    build = _build(
        request=_request(
            selected_endpoints=("tox21",), selected_tox21_tasks=("nr_ar", "sr_mmp")
        )
    )

    async def flaky(*, analysis_id, endpoint, task):
        if task == "nr_ar":
            raise RuntimeError("explainer timeout")
        return _package(endpoint, task)

    async def tox21_snapshot(analysis_id: str, *, session_id: str):
        return SimpleNamespace(**{**vars(SNAPSHOT), "served_endpoints": ("tox21",)})

    _, store, _ = await _run(build, analyses=tox21_snapshot, explanations=flaky)
    refs = _checkpoint(store, BuildStage.GENERATING_EXPLANATIONS).output_refs
    assert set(refs["explanations"]) == {"tox21.sr_mmp"}
    assert set(refs["failed_targets"]) == {"tox21.nr_ar"}


async def test_all_explanations_failing_is_one_gap_not_twelve() -> None:
    async def always_fails(*, analysis_id, endpoint, task):
        raise RuntimeError("explainer down")

    _, store, _ = await _run(_build(), explanations=always_fails)
    checkpoint = _checkpoint(store, BuildStage.GENERATING_EXPLANATIONS)
    assert checkpoint.status is StageStatus.SKIPPED
    assert "no explanation could be produced" in checkpoint.detail


# --- the seam into the fact bundle ------------------------------------------


async def test_the_bundle_is_assembled_from_what_the_stages_checkpointed() -> None:
    _, store, _ = await _run(_build())
    completed = {
        stage: dict(checkpoint.output_refs)
        for stage, checkpoint in read_checkpoints(store.build.stage_state).items()
        if checkpoint.output_refs
    }
    bundle = bundle_from_checkpoints(
        report_build_id=store.build.id,
        analysis_id=ANALYSIS,
        completed=completed,
        predictions=PREDICTIONS,
        explanations=[_package()],
        observation_ids={"herg": OBSERVATION},
    )
    paths = bundle.by_path()
    assert paths["substance.preferred_name"].value == "ethanol"
    assert paths["predictions.herg.probability_blocker"].rendered == "0.027"
    assert paths["explanations.herg.negative_contributor_count"].value == 1
    assert bundle.gaps == ()


async def test_a_skipped_substance_stage_becomes_a_bundle_gap() -> None:
    async def unknown(*, canonical_smiles: str):
        return None

    _, store, _ = await _run(_build(), substances=unknown)
    completed = {
        stage: dict(checkpoint.output_refs)
        for stage, checkpoint in read_checkpoints(store.build.stage_state).items()
        if checkpoint.output_refs
    }
    bundle = bundle_from_checkpoints(
        report_build_id=store.build.id,
        analysis_id=ANALYSIS,
        completed=completed,
        predictions=PREDICTIONS,
    )
    assert any(gap["reason"] == "compound_identity_unresolved" for gap in bundle.gaps)


async def test_a_failed_search_and_a_search_that_never_ran_are_different_gaps() -> None:
    async def down(*, analysis_id, endpoint, compound_names, limit):
        raise ConnectionError("provider down")

    # Provider reached for nothing: the stage declines, and the bundle says the
    # provider was unavailable rather than that the literature is thin.
    _, store, _ = await _run(_build(), evidence=down)
    checkpoint = read_checkpoints(store.build.stage_state)[
        BuildStage.RESEARCHING_EVIDENCE.value
    ]
    assert checkpoint.status is StageStatus.SKIPPED
    assert "could not be reached" in checkpoint.detail

    # Never asked: a different reason entirely.
    build = _build(request=_request(include_external_evidence=False))
    result, _, _ = await _run(build)
    assert {gap["reason"] for gap in result.gaps} == {"external_evidence_not_requested"}


async def test_a_search_that_ran_and_found_nothing_is_recorded_as_having_run() -> None:
    async def empty(*, analysis_id, endpoint, compound_names, limit):
        return []

    _, store, _ = await _run(_build(), evidence=empty)
    refs = _checkpoint(store, BuildStage.RESEARCHING_EVIDENCE).output_refs
    assert refs["search_performed"] is True
    assert refs["promoted_evidence_ids"] == []


# --- recovery ----------------------------------------------------------------


async def test_a_resumed_build_does_not_call_a_provider_twice() -> None:
    calls = {"substance": 0, "evidence": 0}

    async def counting_substance(*, canonical_smiles: str):
        calls["substance"] += 1
        return {"canonical_smiles": canonical_smiles, "preferred_name": "ethanol"}

    async def counting_evidence(*, analysis_id, endpoint, compound_names, limit):
        calls["evidence"] += 1
        return []

    _, store, _ = await _run(
        _build(), substances=counting_substance, evidence=counting_evidence
    )
    await _run(store.build, substances=counting_substance, evidence=counting_evidence)
    assert calls == {"substance": 1, "evidence": 1}


async def test_a_declined_stage_is_not_retried_on_resume() -> None:
    """A build that asked for no evidence does not ask again on recovery."""
    calls = {"n": 0}

    async def counting(*, analysis_id, endpoint, compound_names, limit):
        calls["n"] += 1
        return []

    build = _build(request=_request(include_external_evidence=False))
    _, store, _ = await _run(build, evidence=counting)
    await _run(store.build, evidence=counting)
    assert calls["n"] == 0
